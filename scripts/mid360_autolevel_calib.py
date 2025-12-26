#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MID360 Auto-level calibration (ROS1)
"""
import json
import math
import os
import signal
import time

import rospy
import roslaunch
import rospkg

from sensor_msgs.msg import Imu
from sensor_msgs.msg import PointCloud2

try:
    from livox_ros_driver2.msg import CustomMsg
except Exception:
    CustomMsg = None


def _deg(rad):
    return rad * 180.0 / math.pi


def _norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


def _wait_for_topic_message(topic, msg_type, timeout_s):
    """
    Wait until a topic message is received.
    """
    start = time.time()
    while not rospy.is_shutdown():
        left = timeout_s - (time.time() - start)
        if left <= 0.0:
            raise TimeoutError("timeout waiting topic: %s" % topic)
        try:
            return rospy.wait_for_message(topic, msg_type, timeout=left)
        except rospy.ROSException:
            continue


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _dump_json(path, obj):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def _make_raw_config(base_cfg, lidar_ip=None):
    """
    Copy base config but set extrinsic roll/pitch/yaw to zero for the target lidar.
    """
    cfg = json.loads(json.dumps(base_cfg))
    if "lidar_configs" not in cfg or not cfg["lidar_configs"]:
        raise RuntimeError("invalid config: missing lidar_configs")

    for lc in cfg["lidar_configs"]:
        if lidar_ip is not None and lc.get("ip") != lidar_ip:
            continue
        ext = lc.get("extrinsic_parameter", {})
        ext["roll"] = 0.0
        ext["pitch"] = 0.0
        ext["yaw"] = 0.0
        lc["extrinsic_parameter"] = ext
        if lidar_ip is not None:
            break
    return cfg


def _apply_calib_to_config(base_cfg, roll_deg, pitch_deg, lidar_ip=None):
    """
    Copy base config, only overwrite extrinsic roll/pitch (keep yaw & translation).
    """
    cfg = json.loads(json.dumps(base_cfg))
    if "lidar_configs" not in cfg or not cfg["lidar_configs"]:
        raise RuntimeError("invalid config: missing lidar_configs")

    found = False
    for lc in cfg["lidar_configs"]:
        if lidar_ip is not None and lc.get("ip") != lidar_ip:
            continue
        ext = lc.get("extrinsic_parameter", {})
        # only change roll/pitch; keep yaw and xyz untouched
        ext["roll"] = float(roll_deg)
        ext["pitch"] = float(pitch_deg)
        lc["extrinsic_parameter"] = ext
        found = True
        if lidar_ip is not None:
            break

    if not found:
        raise RuntimeError("cannot find lidar ip=%s in lidar_configs" % str(lidar_ip))
    return cfg


class _ImuAccumulator(object):
    def __init__(self):
        self.n = 0
        self.sum_ax = 0.0
        self.sum_ay = 0.0
        self.sum_az = 0.0
        self.sum_gx = 0.0
        self.sum_gy = 0.0
        self.sum_gz = 0.0
        self.sum_norm_g = 0.0
        self.sum_norm_a = 0.0

    def push(self, msg):
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        gx = msg.angular_velocity.x
        gy = msg.angular_velocity.y
        gz = msg.angular_velocity.z
        self.n += 1
        self.sum_ax += ax
        self.sum_ay += ay
        self.sum_az += az
        self.sum_gx += gx
        self.sum_gy += gy
        self.sum_gz += gz
        self.sum_norm_g += _norm3(gx, gy, gz)
        self.sum_norm_a += _norm3(ax, ay, az)

    def mean(self):
        if self.n <= 0:
            raise RuntimeError("no imu samples")
        inv = 1.0 / float(self.n)
        return {
            "ax": self.sum_ax * inv,
            "ay": self.sum_ay * inv,
            "az": self.sum_az * inv,
            "gx": self.sum_gx * inv,
            "gy": self.sum_gy * inv,
            "gz": self.sum_gz * inv,
            "gyro_norm": self.sum_norm_g * inv,
            "acc_norm": self.sum_norm_a * inv,
            "n": self.n,
        }


class Mid360AutoLevelCalibNode(object):
    def __init__(self):
        self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(self.uuid)
        self.launch = roslaunch.scriptapi.ROSLaunch()
        self.launch.start()

        self.proc_driver = None
        self.proc_rviz = None

        self.imu_topic = rospy.get_param("~imu_topic", "/livox/imu")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/livox/lidar")
        self.check_timeout = float(rospy.get_param("~check_timeout", 8.0))
        self.calib_duration = float(rospy.get_param("~calib_duration", 3.0))
        self.min_samples = int(rospy.get_param("~min_samples", 200))
        self.max_gyro_mean = float(rospy.get_param("~max_gyro_mean", 0.15))  # rad/s
        # Livox IMU 的加速度单位在不同固件/驱动组合中可能是 m/s^2 或 g。
        # 这里默认允许两种范围：约 1g（~1.0）或约 9.81m/s^2。
        self.min_acc_norm = float(rospy.get_param("~min_acc_norm", 0.5))
        self.max_acc_norm = float(rospy.get_param("~max_acc_norm", 15.0))
        self.acc_norm_target_g = float(rospy.get_param("~acc_norm_target_g", 1.0))
        self.acc_norm_target_ms2 = float(rospy.get_param("~acc_norm_target_ms2", 9.81))
        self.acc_norm_tol_g = float(rospy.get_param("~acc_norm_tol_g", 0.5))          # accept [0.5, 1.5] by default
        self.acc_norm_tol_ms2 = float(rospy.get_param("~acc_norm_tol_ms2", 4.0))      # accept [5.81, 13.81] by default

        self.output_config_path = rospy.get_param("~output_config_path", "/tmp/MID360_config_autolevel.json")
        self.raw_config_path = rospy.get_param("~raw_config_path", "/tmp/MID360_config_raw_extrinsic0.json")

        self.start_rviz = bool(rospy.get_param("~start_rviz", False))
        self.rviz_config = rospy.get_param("~rviz_config", "")
        # 仅在“检查 + 标定”阶段静音驱动输出，避免刷屏（日志仍会写入 ROS 日志目录）。
        self.quiet_calib_driver = bool(rospy.get_param("~quiet_calib_driver", True))

        # optional: specify which lidar ip in config to modify (useful for multi-lidar json)
        self.target_lidar_ip = rospy.get_param("~target_lidar_ip", "")
        if self.target_lidar_ip == "":
            self.target_lidar_ip = None

    def _start_driver(self, user_config_path, quiet=False):
        # set global params expected by C++ node (same names as launch files)
        # keep whatever already on param server; only override user_config_path
        rospy.set_param("/user_config_path", user_config_path)

        output_mode = "log" if quiet else "screen"
        node = roslaunch.core.Node(
            package="livox_ros_driver2",
            node_type="livox_ros_driver2_node",
            name="livox_lidar_publisher2",
            output=output_mode,
            respawn=False,
        )
        self.proc_driver = self.launch.launch(node)
        if not self.proc_driver or not self.proc_driver.is_alive():
            raise RuntimeError("failed to start livox_ros_driver2_node via roslaunch python API")

    def _stop_driver(self):
        if self.proc_driver is not None:
            try:
                self.proc_driver.stop()
            except Exception:
                pass
            self.proc_driver = None

    def _start_rviz(self):
        if not self.start_rviz:
            return
        if not self.rviz_config:
            # fall back to package default
            pkg_path = rospkg.RosPack().get_path("livox_ros_driver2")
            self.rviz_config = os.path.join(pkg_path, "config", "display_point_cloud_ROS1.rviz")

        node = roslaunch.core.Node(
            package="rviz",
            node_type="rviz",
            name="livox_rviz",
            output="screen",
            respawn=True,
            args="-d %s" % self.rviz_config,
        )
        self.proc_rviz = self.launch.launch(node)

    def _stop_rviz(self):
        if self.proc_rviz is not None:
            try:
                self.proc_rviz.stop()
            except Exception:
                pass
            self.proc_rviz = None

    def _check_data_ok(self):
        xfer_format = int(rospy.get_param("/xfer_format", 0))
        if xfer_format == 0:
            _wait_for_topic_message(self.lidar_topic, PointCloud2, self.check_timeout)
        else:
            if CustomMsg is None:
                raise RuntimeError("xfer_format != 0 but livox_ros_driver2/CustomMsg not available in python env")
            _wait_for_topic_message(self.lidar_topic, CustomMsg, self.check_timeout)
        _wait_for_topic_message(self.imu_topic, Imu, self.check_timeout)

    def _calibrate_from_imu(self):
        acc = _ImuAccumulator()

        def cb(msg):
            acc.push(msg)

        sub = rospy.Subscriber(self.imu_topic, Imu, cb, queue_size=2000)
        try:
            t0 = time.time()
            rate = rospy.Rate(200)
            while not rospy.is_shutdown() and (time.time() - t0) < self.calib_duration:
                rate.sleep()
            m = acc.mean()
        finally:
            try:
                sub.unregister()
            except Exception:
                pass

        if m["n"] < self.min_samples:
            raise RuntimeError("imu samples too few: %d < %d" % (m["n"], self.min_samples))
        if m["gyro_norm"] > self.max_gyro_mean:
            raise RuntimeError("robot not static? mean gyro norm too large: %.4f rad/s" % m["gyro_norm"])
        acc_norm = m["acc_norm"]
        # 兼容 g 与 m/s^2 两种常见单位；roll/pitch 计算本身与比例无关，所以只做合理性检查。
        ok_g = abs(acc_norm - self.acc_norm_target_g) <= self.acc_norm_tol_g
        ok_ms2 = abs(acc_norm - self.acc_norm_target_ms2) <= self.acc_norm_tol_ms2
        if not (ok_g or ok_ms2):
            raise RuntimeError(
                "acc norm abnormal: %.3f (expected about %.2f±%.2f [g] or %.2f±%.2f [m/s^2])"
                % (acc_norm, self.acc_norm_target_g, self.acc_norm_tol_g, self.acc_norm_target_ms2, self.acc_norm_tol_ms2)
            )

        ax, ay, az = m["ax"], m["ay"], m["az"]
        # classic roll/pitch from accelerometer (ignore yaw)
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.sqrt(ay * ay + az * az))
        return _deg(roll), _deg(pitch), m

    def run(self):
        # resolve base config path
        base_cfg_path = rospy.get_param("/user_config_path", "")
        if not base_cfg_path:
            pkg_path = rospkg.RosPack().get_path("livox_ros_driver2")
            base_cfg_path = os.path.join(pkg_path, "config", "MID360_config.json")

        base_cfg = _load_json(base_cfg_path)

        # 1) start driver with raw extrinsic (RPY=0) for calibration
        raw_cfg = _make_raw_config(base_cfg, lidar_ip=self.target_lidar_ip)
        _dump_json(self.raw_config_path, raw_cfg)

        rospy.loginfo("Starting livox driver for data check/calibration with raw config: %s", self.raw_config_path)
        self._start_driver(self.raw_config_path, quiet=self.quiet_calib_driver)

        # 2) check data OK
        rospy.loginfo("Checking lidar and imu topics are alive: lidar=%s imu=%s", self.lidar_topic, self.imu_topic)
        self._check_data_ok()

        # 3) calib
        rospy.loginfo("Calibrating... keep robot static for %.2f seconds", self.calib_duration)
        roll_deg, pitch_deg, stats = self._calibrate_from_imu()
        rospy.loginfo("Calibration done: roll=%.3f deg pitch=%.3f deg (n=%d, gyro_norm=%.5f, acc_norm=%.3f)",
                      roll_deg, pitch_deg, stats["n"], stats["gyro_norm"], stats["acc_norm"])

        # 4) write final config (only overwrite roll/pitch)
        out_cfg = _apply_calib_to_config(base_cfg, roll_deg, pitch_deg, lidar_ip=self.target_lidar_ip)
        _dump_json(self.output_config_path, out_cfg)
        rospy.loginfo("Wrote calibrated config: %s", self.output_config_path)

        # 5) restart driver with calibrated config
        rospy.loginfo("Restarting livox driver with calibrated config")
        self._stop_driver()
        time.sleep(0.5)
        self._start_driver(self.output_config_path, quiet=False)

        # 6) optionally start RViz
        self._start_rviz()

        rospy.loginfo("Auto-level pipeline finished. Driver is running with calibrated extrinsic. Node will stay alive.")

    def shutdown(self):
        self._stop_rviz()
        self._stop_driver()
        try:
            self.launch.shutdown()
        except Exception:
            pass


def main():
    rospy.init_node("mid360_autolevel_calib", anonymous=False)
    node = Mid360AutoLevelCalibNode()

    def _on_shutdown():
        node.shutdown()

    rospy.on_shutdown(_on_shutdown)
    try:
        node.run()
        rospy.spin()
    except Exception as e:
        rospy.logerr("mid360_autolevel_calib failed: %s", str(e))
        node.shutdown()
        # make roslaunch stop (required=true will shutdown the launch)
        os.kill(os.getpid(), signal.SIGINT)


if __name__ == "__main__":
    main()


