import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import math
from collections import deque
import tkinter as tk
from tkinter import filedialog, messagebox

from vispy import scene, app
from vispy.scene import visuals
from vispy.color import get_colormap

class WaterfallCanvas(scene.SceneCanvas):
    def __init__(self, samplerate, n_fft, audio_info_text, **kwargs):
        super(WaterfallCanvas, self).__init__(keys='interactive', size=(1280, 720),
                                              title='3D Audio Waterfall', **kwargs)
        self.unfreeze()
        self.samplerate = samplerate
        self.n_fft = n_fft
        self.x_scale = 20.0 / (samplerate/2)
        self.freqs_Hz = np.linspace(0, samplerate/2, n_fft)
        self.freqs = self.freqs_Hz * self.x_scale
        self.slices = deque(maxlen=100)
        self.slice_colors = deque(maxlen=100)
        self.new_slice_queue = deque()
        self.speed = 5.0
        self.amp_scale = 0.05
        self.cmap = get_colormap('viridis')

        self.view = self.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(elevation=30, azimuth=-90,
                                                         distance=30, fov=45)
        self.view.camera.center = (10, 0, 5)
        self.view.camera.up = 'y'
        self.view.scene.transform = scene.transforms.MatrixTransform()
        self.view.scene.transform.rotate(60, (0, 1, 0))

        self.mesh = visuals.Mesh()
        self.view.add(self.mesh)

        self.text = scene.visuals.Text(audio_info_text, color='white', pos=(10, 10),
                                       font_size=12, anchor_x='left', anchor_y='bottom')
        self.text.transform = scene.transforms.STTransform(translate=(10, self.size[1]-20))
        self.text.parent = self.scene

        self._timer = app.Timer(interval=1/60.0, connect=self.on_timer, start=True)
        self.freeze()

    def on_timer(self, event):
        dt = event.dt
        for i in range(len(self.slices)):
            self.slices[i][:, 2] += self.speed * dt

        while self.new_slice_queue:
            fft_db = self.new_slice_queue.popleft()
            vertices = np.column_stack((self.freqs, fft_db * self.amp_scale, np.zeros_like(self.freqs)))
            norm = np.clip((fft_db + 100) / 100, 0, 1)
            colors = self.cmap.map(norm)
            self.slices.append(vertices)
            self.slice_colors.append(colors)

        if self.slices:
            full_vertices = np.vstack(self.slices)
            full_colors = np.vstack(self.slice_colors)
            self.mesh.set_data(vertices=full_vertices, color=full_colors)
        
        self.update()

def audio_thread_func(file_path, canvas, desired_blocks_per_sec=50):
    try:
        file = sf.SoundFile(file_path)
    except Exception as e:
        print("Error opening audio file:", e)
        return

    samplerate = file.samplerate
    raw_block = samplerate / desired_blocks_per_sec
    blocksize = int(2 ** math.ceil(math.log(raw_block, 2)))
    n_fft = blocksize // 2 + 1

    def callback(outdata, frames, time_info, status):
        if status:
            print("Audio callback status:", status)
        data = file.read(frames)
        n_read = len(data)
        if n_read < frames:
            outdata[:n_read] = data
            outdata[n_read:] = 0
            raise sd.CallbackStop
        else:
            outdata[:] = data
        channel_data = data[:, 0]
        window = np.hanning(len(channel_data))
        channel_data = channel_data * window
        fft_result = np.abs(np.fft.rfft(channel_data))
        fft_db = 20 * np.log10(fft_result + 1e-6)
        canvas.new_slice_queue.append(fft_db)

    with sd.OutputStream(samplerate=samplerate, channels=file.channels,
                         callback=callback, blocksize=blocksize):
        duration_ms = int(file.frames / samplerate * 1000)
        sd.sleep(duration_ms + 1000)
    file.close()

def main():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="选择音频文件",
                                           filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")])
    if not file_path:
        messagebox.showerror("错误", "未选择音频文件！")
        return

    try:
        info = sf.info(file_path)
    except Exception as e:
        messagebox.showerror("错误", f"无法获取音频文件信息:\n{e}")
        return

    audio_info_text = (f"文件: {file_path}\n"
                       f"格式: {info.format}\n"
                       f"子类型: {info.subtype}\n"
                       f"采样率: {info.samplerate} Hz\n"
                       f"声道数: {info.channels}\n"
                       f"帧数: {info.frames}\n"
                       f"时长: {info.duration:.2f} s")

    samplerate = info.samplerate
    desired_blocks_per_sec = 50
    raw_block = samplerate / desired_blocks_per_sec
    blocksize = int(2 ** math.ceil(math.log(raw_block, 2)))
    n_fft = blocksize // 2 + 1

    canvas = WaterfallCanvas(samplerate, n_fft, audio_info_text)
    canvas.show()

    audio_thread = threading.Thread(target=audio_thread_func,
                                    args=(file_path, canvas, desired_blocks_per_sec),
                                    daemon=True)
    audio_thread.start()

    app.run()

if __name__ == '__main__':
    main()
