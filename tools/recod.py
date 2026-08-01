from datetime import timedelta
from pathlib import Path
import time

import dv_processing as dv


def record() -> None:
    output_path = Path("davis_events_frames.aedat4").resolve()

    # 打开当前连接的第一台 DAVIS 相机
    capture = dv.io.camera.DAVIS()

    # 同时启用事件和 APS/RGB 帧
    capture.setEventsRunning(True)
    capture.setFramesRunning(True)

    # 约 30 FPS；可根据需要修改
    capture.setFrameInterval(timedelta(milliseconds=33))

    # 可选择自动曝光
    capture.setAutoExposure(True)

    # 根据相机支持的数据流自动创建 AEDAT4 输出
    writer = dv.io.MonoCameraWriter(str(output_path), capture)

    print(f"Camera: {capture.getCameraName()}")
    print(f"Recording to: {output_path}")
    print("Press Ctrl+C to stop recording.")

    try:
        while capture.isRunning():
            received_data = False

            # 读取并写入原始事件
            events = capture.getNextEventBatch()
            if events is not None and events.size() > 0:
                writer.writeEvents(events)
                received_data = True

            # 读取并写入 APS/RGB 帧
            frame = capture.getNextFrame()
            if frame is not None:
                writer.writeFrame(frame)
                received_data = True

            if not received_data:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nStopping recording...")

    finally:
        # MonoCameraWriter 析构时完成文件刷新和关闭
        del writer
        del capture

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    record()