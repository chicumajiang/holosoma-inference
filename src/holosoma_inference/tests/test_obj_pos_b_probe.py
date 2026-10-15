import json
import sys
import time
import types
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src/holosoma_inference"), str(ROOT / "src/holosoma")]

import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from holosoma_inference.test_obj_pos_b import ObjPosProbe, ProbeOptions, Rates, relative_positions, ObservationInterface
from holosoma_inference.deploy_tri_policy import Tracking, torso_from_pelvis


class Binding:
    def __init__(self):
        self.tick = 1
        self.q = np.zeros(29)
        self.freeze = False
        self.mode_calls = self.writes = 0

    def read_low_state(self):
        if not self.freeze:
            self.tick += 1
        return types.SimpleNamespace(tick=self.tick,
            motor_state=[types.SimpleNamespace(q=float(q), dq=0.0) for q in self.q],
            imu_state=types.SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0]))

    def set_control_mode(self, mode):
        self.mode_calls += 1

    def write_low_command(self, cmd):
        self.writes += 1


def message(pos, stamp=None, frame="world", quat=(0., 0., 0., 1.)):
    msg = PoseStamped()
    msg.header.frame_id = frame
    stamp = time.time() if stamp is None else stamp
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = [float(x) for x in pos]
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = quat
    return msg


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.binding = Binding()
        class Subscriber:
            def __init__(self, topic, kind):
                assert topic == "rt/lowstate"

            def Init(self, callback, depth):
                self.callback = callback

            def Close(self):
                pass

        self.factory = patch("unitree_sdk2py.core.channel.ChannelFactoryInitialize")
        self.subscriber = patch("unitree_sdk2py.core.channel.ChannelSubscriber", Subscriber)
        self.publisher = patch("unitree_sdk2py.core.channel.ChannelPublisher", side_effect=AssertionError("No publisher allowed"))
        original_get = ObservationInterface.get_low_state
        binding = self.binding

        def get_sample(interface):
            interface._callback(binding.read_low_state())
            return original_get(interface)

        self.sdk_get = patch.object(ObservationInterface, "get_low_state", get_sample)
        self.native_factory = patch("unitree_interface.create_robot", side_effect=AssertionError("No bidirectional binding factory"))
        self.onnx_session = patch("onnxruntime.InferenceSession", side_effect=AssertionError("No ONNX session"))
        for p in (self.factory, self.subscriber, self.publisher, self.sdk_get, self.native_factory, self.onnx_session):
            p.start()
        self.probe = ObjPosProbe(ProbeOptions(interface="fake_eth", mocap_unit="m"))

    def tearDown(self):
        self.probe.close()
        for p in (self.factory, self.subscriber, self.publisher, self.sdk_get, self.native_factory, self.onnx_session):
            p.stop()
        self.assertEqual(self.binding.mode_calls, 0)
        self.assertEqual(self.binding.writes, 0)

    def seed(self, skew=0, frame="world"):
        self.probe.poll_sdk()
        self.probe.poll_sdk()
        t = time.time()
        self.probe.pelvis._callback(message([1, 2, 0.8], t))
        self.probe.ball._callback(message([1.3960365, 1.9, 0.244], t + skew, frame))
        self.probe.poll_sdk()

    def test_geometry_zero_waist_and_shared_deployment_fk(self):
        self.seed()
        result = self.probe.compute()
        self.assertTrue(result["valid"], result["reasons"])
        np.testing.assert_allclose(result["obj_pos_b"], [.4, -.1, -.6], atol=1e-6)
        np.testing.assert_allclose(result["ball_pos_pelvis"], [.3960365, -.1, -.556], atol=1e-6)
        state = self.probe.states[-1]["state"]
        fake_tracker = types.SimpleNamespace(pelvis=self.probe.pelvis, require=lambda: None)
        deployed = Tracking.torso_pose(fake_tracker, state, self.probe.robot)
        expected = torso_from_pelvis(state, self.probe.robot, self.probe.pelvis.sample)
        np.testing.assert_allclose(deployed[0], expected[0])
        np.testing.assert_allclose(deployed[1], expected[1])

    def test_nonzero_waist_fk_matches_manual_chain(self):
        self.seed()
        state = self.probe.states[-1]["state"].copy()
        a, b, c = .3, -.2, .4
        state[0, 19:22] = [a, b, c]
        rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        ry = np.array([[np.cos(c), 0, np.sin(c)], [0, 1, 0], [-np.sin(c), 0, np.cos(c)]])
        p = self.probe.pelvis.sample
        ball = self.probe.ball.sample
        result = relative_positions(state, self.probe.robot, p, ball)
        torso = p["position"] + rz @ [-.0039635, 0, .044]
        np.testing.assert_allclose(result["torso_pos_w"], torso, atol=1e-6)
        np.testing.assert_allclose(result["obj_pos_b"], (rz @ rx @ ry).T @ (ball["position"] - torso), atol=1e-6)

    def test_global_translation_and_rotation_do_not_change_relative_position(self):
        self.seed()
        state = self.probe.states[-1]["state"]
        p, b = self.probe.pelvis.sample, self.probe.ball.sample
        original = relative_positions(state, self.probe.robot, p, b)
        angle = .7
        r = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        shift = np.array([10, -7, 2.0])
        rotated_p = dict(p, position=r @ p["position"] + shift, quat_xyzw=np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)]))
        rotated_b = dict(b, position=r @ b["position"] + shift)
        result = relative_positions(state, self.probe.robot, rotated_p, rotated_b)
        for key in ("obj_pos_b", "ball_pos_pelvis"):
            np.testing.assert_allclose(result[key], original[key], atol=1e-6)

    def test_wrong_frame_and_skew_are_marked_invalid(self):
        self.seed(skew=.1, frame="other")
        result = self.probe.compute()
        self.assertFalse(result["valid"])
        self.assertIsNone(result["obj_pos_b"])
        self.assertIn("global frame_id 不一致或为空", result["reasons"])
        self.assertIn("动捕时间差过大", result["reasons"])

    def test_stopped_messages_invalidate_last_good_result(self):
        self.seed()
        self.assertTrue(self.probe.compute()["valid"])
        for buffer in self.probe.poses.values():
            for sample in buffer:
                sample["received_at"] -= 1
        self.assertFalse(self.probe.compute()["valid"])

    def test_frequency_counts_source_and_receive_separately(self):
        stats = Rates()
        for i in range(101):
            stats.observe(i * .01, 123 + i * .02)
        summary = stats.summary(1.0)
        self.assertAlmostEqual(summary["receive_hz"], 100)
        self.assertAlmostEqual(summary["source_hz"], 50)
        stats.observe(1.1, 125)
        self.assertEqual(stats.duplicates, 1)
        self.assertEqual(stats.summary(3)["receive_hz"], 0)

    def test_recording_contains_raw_sdk_stage_and_computation(self):
        self.probe.handle_key("r")
        self.seed()
        self.probe.step()
        self.probe.handle_key("2")
        invalid = self.probe.step()
        self.assertFalse(invalid["valid"])
        self.assertIsNone(invalid["obj_pos_b"])
        self.seed()
        self.probe.step()
        self.probe.handle_key("3")
        self.seed()
        self.probe.step()
        self.probe.handle_key("4")
        self.seed()
        self.probe.step()
        path = self.probe.path
        self.probe.handle_key("r")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(rows[0]["type"], "metadata")
        for kind in ("pose", "sdk", "calculation", "stage_event", "summary"):
            self.assertTrue(any(row["type"] == kind for row in rows), kind)
        self.assertTrue(any(row.get("valid") and row["type"] == "calculation" for row in rows))
        self.assertEqual(len(rows[-1]["segments"]), 4)
        self.assertEqual(self.binding.writes, 0)
        print("PROBE_TEST_RECORD", path)

    def test_command_methods_raise_even_if_write_flag_is_changed(self):
        self.probe.interface.write_enabled = True
        with self.assertRaises(RuntimeError):
            self.probe.interface.send_low_command(np.zeros(29), np.zeros(29), np.zeros(29))
        with self.assertRaises(RuntimeError):
            self.probe.interface._send_binding_command(np.zeros(29), np.zeros(29), np.zeros(29))

    def test_units_scale_correctly(self):
        t = time.time()
        for unit, raw in (("mm", [1000, 2000, 800]), ("cm", [100, 200, 80]), ("m", [1, 2, .8])):
            self.probe.pelvis.scale = {"mm": .001, "cm": .01, "m": 1}[unit]
            t += .01
            self.probe.pelvis._callback(message(raw, t))
            np.testing.assert_allclose(self.probe.pelvis.sample["position"], [1, 2, .8])

    def test_three_axis_ball_motion_has_correct_sign_and_no_cross_axis_leakage(self):
        self.seed()
        state = self.probe.states[-1]["state"]
        pelvis, ball = self.probe.pelvis.sample, self.probe.ball.sample
        baseline = relative_positions(state, self.probe.robot, pelvis, ball)
        for axis in range(3):
            delta = np.eye(3)[axis] * 0.1
            moved = relative_positions(state, self.probe.robot, pelvis, dict(ball, position=ball["position"] + delta))
            np.testing.assert_allclose(moved["ball_pos_pelvis"] - baseline["ball_pos_pelvis"], delta, atol=1e-6)
            np.testing.assert_allclose(moved["obj_pos_b"] - baseline["obj_pos_b"], delta, atol=1e-6)

    def test_tick_stall_does_not_retimestamp_old_joint_state(self):
        self.seed()
        self.probe.step()
        self.binding.freeze = True
        self.probe.interface.last_tick_at -= .1
        stamp = time.time()
        self.probe.pelvis._callback(message([1, 2, .8], stamp))
        self.probe.ball._callback(message([1.3960365, 1.9, .244], stamp))
        self.probe.states.clear()
        self.probe.poll_sdk()
        result = self.probe.compute()
        self.assertFalse(result["valid"])
        self.assertGreater(result["joint_receive_skew"], self.probe.options.max_joint_skew)

    def test_no_sdk_sample_returns_null_instead_of_crashing(self):
        self.seed()
        self.probe.states.clear()
        result = self.probe.compute()
        self.assertFalse(result["valid"])
        self.assertIsNone(result["obj_pos_b"])

    def test_nearest_invalid_sample_is_not_bypassed(self):
        self.seed()
        self.probe.states.clear()
        midpoint = (self.probe.poses["pelvis"][-1]["received_at"] + self.probe.poses["ball"][-1]["received_at"]) / 2
        state = np.zeros((1, 71))
        self.probe.states.append(dict(sequence=1, received_at=midpoint - .02, observed_update_at=midpoint - .02, state=state, valid=True))
        self.probe.states.append(dict(sequence=2, received_at=midpoint, observed_update_at=midpoint, state=state, valid=False))
        self.probe.states.append(dict(sequence=3, received_at=midpoint + .02, observed_update_at=midpoint + .02, state=state, valid=True))
        result = self.probe.compute()
        self.assertEqual(result["sdk_sequence"], 2)
        self.assertFalse(result["valid"])

    def test_pair_can_recover_after_invalid_sdk_without_counting_twice(self):
        self.seed()
        with patch.object(self.probe, "poll_sdk", return_value=None):
            for s in self.probe.states:
                s["valid"] = False
            self.assertFalse(self.probe.step()["valid"])
            for s in self.probe.states:
                s["valid"] = True
            self.assertTrue(self.probe.step()["valid"])
            count = self.probe.rates["valid_computations"].count
            self.probe.states[-1]["valid"] = False
            self.assertFalse(self.probe.step()["valid"])
            self.probe.states[-1]["valid"] = True
            self.assertTrue(self.probe.step()["valid"])
            self.assertEqual(self.probe.rates["valid_computations"].count, count)

    def test_real_ros_callbacks_record_with_fake_sdk(self):
        import os
        prefix = f"/probe_geometry_test_{os.getpid()}"
        options = replace(ProbeOptions(), interface="fake_eth", mocap_unit="m", pelvis_topic=prefix + "/pelvis",
                          ball_topic=prefix + "/ball", max_seconds=1.5, record_on_start=True)
        probe = ObjPosProbe(options)
        publisher = Node(f"probe_synthetic_publisher_{os.getpid()}")
        pp = publisher.create_publisher(PoseStamped, options.pelvis_topic, 10)
        bp = publisher.create_publisher(PoseStamped, options.ball_topic, 10)
        stop = threading.Event()

        def publish():
            while not stop.is_set():
                stamp = time.time()
                pp.publish(message([1, 2, .8], stamp))
                bp.publish(message([1.3960365, 1.9, .244], stamp))
                stop.wait(.02)

        thread = threading.Thread(target=publish)
        thread.start()
        try:
            with patch("sys.stdin.isatty", return_value=False):
                probe.run()
            rows = [json.loads(line) for line in probe.path.read_text().splitlines()]
            valid = [r for r in rows if r["type"] == "calculation" and r["valid"]]
            self.assertGreater(len(valid), 3)
            np.testing.assert_allclose(valid[-1]["obj_pos_b"], [.4, -.1, -.6], atol=1e-6)
            print("ROS_INTEGRATION_RECORD", probe.path)
        finally:
            stop.set()
            thread.join(timeout=2)
            probe.close()
            publisher.destroy_node()

    def test_pre_phase_inflight_sdk_sample_is_not_admitted(self):
        self.probe.handle_key("r")
        self.seed()
        sample = self.probe.states[-1]
        self.probe.handle_key("2")
        self.probe.capture_sdk(sample)
        self.assertEqual(len(self.probe.states), 0)
        self.assertEqual(self.probe.current_segment["rates"]["sdk_receive"].count, 0)
        path = self.probe.path
        self.probe.handle_key("r")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertTrue(any(row["type"] == "sdk" and not row["admitted_to_phase"] for row in rows))


if __name__ == "__main__":
    unittest.main()
