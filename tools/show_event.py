from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
import dv_processing as dv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="实时可视化 DAVIS 的 APS 图像和事件流。"
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="相机序列号，例如 00000055；默认打开第一台相机。",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=33,
        help="每张事件可视化图累积的时间窗口，单位 ms，默认 33。",
    )
    parser.add_argument(
        "--display-fps",
        type=float,
        default=30.0,
        help="预览刷新率，默认 30 FPS。",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="可选：保存并排预览视频，例如 preview.mp4。",
    )
    return parser.parse_args()


def image_to_bgr_u8(image: np.ndarray) -> np.ndarray:
    """将 DAVIS 帧转换为 OpenCV 可显示的 uint8 BGR 图像。"""
    image = np.asarray(image)

    if image.dtype != np.uint8:
        min_value = float(np.min(image))
        max_value = float(np.max(image))

        if max_value > min_value:
            image = cv.normalize(
                image,
                None,
                0,
                255,
                cv.NORM_MINMAX,
            ).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)

    if image.ndim == 2:
        return cv.cvtColor(image, cv.COLOR_GRAY2BGR)

    if image.ndim == 3:
        if image.shape[2] == 1:
            return cv.cvtColor(image, cv.COLOR_GRAY2BGR)

        if image.shape[2] == 3:
            return image.copy()

        if image.shape[2] == 4:
            return cv.cvtColor(image, cv.COLOR_BGRA2BGR)

    raise ValueError(f"不支持的图像尺寸：{image.shape}")


def add_panel_label(
    image: np.ndarray,
    title: str,
    subtitle: str = "",
) -> np.ndarray:
    result = image.copy()

    cv.rectangle(
        result,
        (0, 0),
        (result.shape[1], 52),
        (0, 0, 0),
        thickness=-1,
    )

    cv.putText(
        result,
        title,
        (10, 22),
        cv.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv.LINE_AA,
    )

    if subtitle:
        cv.putText(
            result,
            subtitle,
            (10, 44),
            cv.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv.LINE_AA,
        )

    return result


def open_camera(serial: Optional[str]):
    cameras = dv.io.camera.discover()

    print(f"检测到 {len(cameras)} 台相机：")
    for camera in cameras:
        print(" ", camera)

    if not cameras:
        raise RuntimeError("没有检测到 iniVation 相机。")

    if serial:
        print(f"尝试打开序列号：{serial}")
        return dv.io.camera.open(serial)

    print("尝试打开第一台相机。")
    return dv.io.camera.open()


def main() -> int:
    args = parse_args()

    if args.window_ms <= 0:
        raise ValueError("--window-ms 必须大于 0。")

    if args.display_fps <= 0:
        raise ValueError("--display-fps 必须大于 0。")

    capture = None
    video_writer = None

    try:
        capture = open_camera(args.serial)

        camera_name = capture.getCameraName()
        print("相机已打开：", camera_name)

        if not capture.isEventStreamAvailable():
            raise RuntimeError("当前相机没有事件流输出。")

        has_frames = capture.isFrameStreamAvailable()

        print("事件流：可用")
        print("APS 帧：", "可用" if has_frames else "不可用")

        event_width, event_height = capture.getEventResolution()

        # 官方事件流可视化器
        visualizer = dv.visualization.EventVisualizer(
            capture.getEventResolution()
        )

        # 设置成白底、正负事件分色；某些旧版本若不支持则跳过
        try:
            visualizer.setBackgroundColor(
                dv.visualization.colors.white()
            )
            visualizer.setPositiveColor(
                dv.visualization.colors.iniBlue()
            )
            visualizer.setNegativeColor(
                dv.visualization.colors.darkGray()
            )
        except (AttributeError, TypeError):
            print("当前 dv-processing 版本不支持颜色设置，使用默认配色。")

        latest_event_image = np.full(
            (event_height, event_width, 3),
            255,
            dtype=np.uint8,
        )
        latest_rgb_image = np.zeros(
            (event_height, event_width, 3),
            dtype=np.uint8,
        )

        event_count = 0
        latest_frame_timestamp = None

        slicer = dv.EventStreamSlicer()

        def event_callback(events: dv.EventStore) -> None:
            nonlocal latest_event_image, event_count

            latest_event_image = visualizer.generateImage(events)
            latest_event_image = image_to_bgr_u8(latest_event_image)
            event_count = int(events.size())

        slicer.doEveryTimeInterval(
            timedelta(milliseconds=args.window_ms),
            event_callback,
        )

        window_name = "DAVIS RGB + Events"
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        display_interval = 1.0 / args.display_fps
        last_display_time = 0.0

        print()
        print("开始预览。")
        print("按 Q 或 Esc 退出。")
        if args.save is not None:
            print("可视化视频将保存到：", args.save.resolve())

        while capture.isRunning():
            received_data = False

            events = capture.getNextEventBatch()
            if events is not None and events.size() > 0:
                slicer.accept(events)
                received_data = True

            if has_frames:
                frame = capture.getNextFrame()

                if frame is not None:
                    latest_rgb_image = image_to_bgr_u8(frame.image)
                    latest_frame_timestamp = getattr(
                        frame,
                        "timestamp",
                        None,
                    )
                    received_data = True

            current_time = time.perf_counter()

            if current_time - last_display_time >= display_interval:
                last_display_time = current_time

                event_panel = cv.resize(
                    latest_event_image,
                    (event_width, event_height),
                    interpolation=cv.INTER_NEAREST,
                )

                rgb_panel = cv.resize(
                    latest_rgb_image,
                    (event_width, event_height),
                    interpolation=cv.INTER_AREA,
                )

                rgb_subtitle = "APS frame"
                if latest_frame_timestamp is not None:
                    rgb_subtitle += f" | t={latest_frame_timestamp} us"

                event_subtitle = (
                    f"{event_count} events / "
                    f"{args.window_ms} ms"
                )

                rgb_panel = add_panel_label(
                    rgb_panel,
                    "APS / RGB",
                    rgb_subtitle,
                )
                event_panel = add_panel_label(
                    event_panel,
                    "Events",
                    event_subtitle,
                )

                canvas = np.hstack((rgb_panel, event_panel))

                cv.imshow(window_name, canvas)

                if args.save is not None:
                    if video_writer is None:
                        args.save.parent.mkdir(
                            parents=True,
                            exist_ok=True,
                        )

                        fourcc = cv.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv.VideoWriter(
                            str(args.save),
                            fourcc,
                            args.display_fps,
                            (canvas.shape[1], canvas.shape[0]),
                        )

                        if not video_writer.isOpened():
                            raise RuntimeError(
                                f"无法创建视频文件：{args.save}"
                            )

                    video_writer.write(canvas)

                key = cv.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

            if not received_data:
                time.sleep(0.001)

    except RuntimeError as error:
        message = str(error)

        print("\n相机读取失败：", file=sys.stderr)
        print(message, file=sys.stderr)

        if "Camera time synchronization timeout" in message:
            print(
                "\n这是相机初始化阶段的时间戳同步失败，"
                "不是 OpenCV 可视化代码导致的。",
                file=sys.stderr,
            )
            print(
                "请完全退出 DV Viewer 和 dv-runtime、拔插相机，"
                "再单独运行本脚本。",
                file=sys.stderr,
            )

        return 1

    except KeyboardInterrupt:
        print("\n用户中断。")
        return 0

    finally:
        if video_writer is not None:
            video_writer.release()

        cv.destroyAllWindows()

        if capture is not None:
            del capture

    print("预览结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())