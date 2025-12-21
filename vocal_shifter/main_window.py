import sys
import os
import time
import json
import pathlib
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QComboBox, QDoubleSpinBox, QSpinBox)
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

# Import widgets
from .widgets import CustomViewBox, PianoRollAxis, BPMAxis
# Import AudioProcessor
from .audio_processor import AudioProcessor
# Import Config Manager
from . import config_manager

class VocalShifterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SingingVocoders - 音高修正工具")
        self.resize(1200, 800)
        
        # Initialize Audio Processor
        self.processor = AudioProcessor()
        
        # Data for UI
        self.project_path = None
        self.audio_path = None
        self.model_path = None
        self.f0 = None # numpy array (current editing)
        self.f0_original = None # numpy array (reference)
        self.waveform_x = None
        self.waveform_y = None
        self.synthesized_audio = None
        self.is_dirty = False # Flag to track if F0 has changed since last synthesis
        self.last_shift_value = 0.0
        
        # Segment State
        self.segment_states = [] # List of dicts: {'dirty': bool, 'audio': np.array}
        
        # Playback State
        self.is_playing = False
        self.current_playback_time = 0.0 # seconds
        self.playback_start_time = 0.0 # seconds (for return to start)
        self.last_wall_time = 0.0
        self.playback_timer = QTimer()
        self.playback_timer.setInterval(30) # 30ms update
        self.playback_timer.timeout.connect(self.update_cursor)
        
        # Undo/Redo Stacks
        self.undo_stack = []
        self.redo_stack = []
        
        # Interaction State
        self.is_drawing = False
        self.last_mouse_pos = None
        
        self.init_ui()
        self.load_default_model()
        
    def create_menu_bar(self):
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("文件")
        
        open_action = QAction("打开工程", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.open_project_dialog)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存工程", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("另存为工程", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()

        load_model_action = QAction("加载模型", self)
        load_model_action.triggered.connect(self.load_model_dialog)
        file_menu.addAction(load_model_action)

        load_audio_action = QAction("加载音频", self)
        load_audio_action.triggered.connect(self.load_audio_dialog)
        file_menu.addAction(load_audio_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("导出音频", self)
        export_action.triggered.connect(self.export_audio_dialog)
        file_menu.addAction(export_action)

        # Edit Menu
        edit_menu = menu_bar.addMenu("编辑")
        
        undo_action = QAction("撤销", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("重做", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.redo)
        edit_menu.addAction(redo_action)

        # Playback Menu
        play_menu = menu_bar.addMenu("播放")
        
        play_orig_action = QAction("播放原音", self)
        play_orig_action.triggered.connect(self.play_original)
        play_menu.addAction(play_orig_action)
        
        synth_play_action = QAction("合成并播放", self)
        synth_play_action.triggered.connect(self.synthesize_and_play)
        play_menu.addAction(synth_play_action)
        
        stop_action = QAction("停止", self)
        stop_action.setShortcut(Qt.Key.Key_Escape)
        stop_action.triggered.connect(self.stop_audio)
        play_menu.addAction(stop_action)

        # Settings Menu
        settings_menu = menu_bar.addMenu("设置")
        
        set_default_model_action = QAction("设置默认模型", self)
        set_default_model_action.triggered.connect(self.set_default_model_dialog)
        settings_menu.addAction(set_default_model_action)

    def init_ui(self):
        self.create_menu_bar()
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Controls Bar
        controls_layout = QHBoxLayout()
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        controls_layout.setContentsMargins(10, 5, 10, 5)

        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(-24, 24)
        self.shift_spin.setSingleStep(1)
        self.shift_spin.setPrefix("移调: ")
        self.shift_spin.setSuffix(" 半音")
        self.shift_spin.valueChanged.connect(self.apply_shift)
        
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setRange(10, 300)
        self.bpm_spin.setValue(120)
        self.bpm_spin.setPrefix("BPM: ")
        self.bpm_spin.valueChanged.connect(lambda: self.plot_widget.getAxis('bottom').update())

        self.beats_spin = QSpinBox()
        self.beats_spin.setRange(1, 32)
        self.beats_spin.setValue(4)
        self.beats_spin.setPrefix("拍号: ")
        self.beats_spin.setSuffix(" / 4")
        self.beats_spin.valueChanged.connect(lambda: self.plot_widget.getAxis('bottom').update())

        controls_layout.addWidget(QLabel("参数设置:"))
        controls_layout.addWidget(self.shift_spin)
        controls_layout.addWidget(self.bpm_spin)
        controls_layout.addWidget(self.beats_spin)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Plot Area
        self.plot_widget = pg.PlotWidget(
            viewBox=CustomViewBox(self), 
            axisItems={
                'left': PianoRollAxis(orientation='left'),
                'top': BPMAxis(self, orientation='top'),
                'bottom': pg.AxisItem(orientation='bottom') # Standard axis, will be hidden
            }
        )
        self.plot_widget.setBackground('#2b2b2b')
        self.plot_widget.setLabel('left', '音高 (Note)')
        # self.plot_widget.setLabel('bottom', '小节-拍') # Moved to top
        self.plot_widget.showGrid(x=True, y=True, alpha=0.5)
        self.plot_widget.getAxis('left').setGrid(128)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        # Configure Axes
        self.plot_widget.showAxis('top')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.getAxis('top').setLabel('小节-拍')
        
        # Set Limits
        self.plot_widget.plotItem.vb.setLimits(xMin=0)
        
        # Playback Cursor
        self.play_cursor = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('y', width=2), label='Play')
        self.plot_widget.addItem(self.play_cursor)
        
        # Waveform View
        self.waveform_view = pg.ViewBox()
        self.waveform_view.setMouseEnabled(x=False, y=False)
        self.waveform_view.setMenuEnabled(False)
        self.waveform_view.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.waveform_view.setXLink(self.plot_widget.plotItem.vb)
        self.waveform_view.setYRange(-1, 1)
        self.waveform_view.setZValue(-1)
        self.plot_widget.scene().addItem(self.waveform_view)
        
        self.plot_widget.plotItem.vb.sigResized.connect(self.update_views)

        # Custom Mouse Interaction
        self.plot_widget.scene().sigMouseMoved.connect(self.on_scene_mouse_move)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_scene_mouse_click)
        
        # Curves
        self.waveform_curve = pg.PlotCurveItem(pen=pg.mkPen(color=(255, 255, 255, 30), width=1), name="Waveform")
        self.waveform_view.addItem(self.waveform_curve)
        
        self.f0_orig_curve_item = self.plot_widget.plot(pen=pg.mkPen(color=(255, 255, 255, 80), width=2, style=Qt.PenStyle.DashLine), name="Original F0")
        self.f0_curve_item = self.plot_widget.plot(pen=pg.mkPen('#00ff00', width=3), name="F0")
        
        layout.addWidget(self.plot_widget)
        
        # Instructions
        instructions = QLabel("使用说明: 加载模型 -> 加载音频 -> 左键绘制音高，右键还原音高，中键拖动，Ctrl+滚轮缩放X，Alt+滚轮缩放Y -> 合成")
        layout.addWidget(instructions)
        
        # Status
        self.status_label = QLabel("就绪")
        layout.addWidget(self.status_label)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self.toggle_playback()
        elif ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if ev.key() == Qt.Key.Key_Z:
                if ev.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self.redo()
                else:
                    self.undo()
            elif ev.key() == Qt.Key.Key_Y:
                self.redo()
        else:
            super().keyPressEvent(ev)

    def push_undo(self):
        if self.f0 is None: return
        # Deep copy current state
        self.undo_stack.append(self.f0.copy())
        # Limit stack size
        if len(self.undo_stack) > 16:
            self.undo_stack.pop(0)
        # Clear redo stack on new action
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            self.status_label.setText("没有可撤销的操作")
            return
            
        # Save current state to redo
        self.redo_stack.append(self.f0.copy())
        
        # Restore from undo
        self.f0 = self.undo_stack.pop()
        self.is_dirty = True
        self.update_plot()
        self.status_label.setText("撤销")

    def redo(self):
        if not self.redo_stack:
            self.status_label.setText("没有可重做的操作")
            return
            
        # Save current state to undo
        self.undo_stack.append(self.f0.copy())
        
        # Restore from redo
        self.f0 = self.redo_stack.pop()
        self.is_dirty = True
        self.update_plot()
        self.status_label.setText("重做")

    def toggle_playback(self):
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def start_playback(self):
        # Auto-synthesize if needed
        if self.is_dirty or self.synthesized_audio is None:
            if self.processor.model is not None and self.processor.audio is not None:
                self.synthesize_audio_only()
            elif self.synthesized_audio is None:
                return

        try:
            if not self.is_playing:
                self.playback_start_time = self.current_playback_time

            sr = self.processor.sr
            total_duration = len(self.synthesized_audio) / sr
            
            if self.current_playback_time >= total_duration:
                self.current_playback_time = 0
                self.play_cursor.setValue(0)
            
            start_sample = int(self.current_playback_time * sr)
            audio_to_play = self.synthesized_audio[start_sample:]
            
            if len(audio_to_play) == 0:
                self.current_playback_time = 0
                self.play_cursor.setValue(0)
                audio_to_play = self.synthesized_audio
            
            sd.play(audio_to_play, sr)
            self.is_playing = True
            self.last_wall_time = time.time()
            self.playback_timer.start()
            self.status_label.setText("正在播放...")
            
        except Exception as e:
            print(f"Playback error: {e}")
            self.stop_playback()

    def synthesize_audio_only(self):
        """Helper to synthesize without playing immediately (used by start_playback)"""
        try:
            self.status_label.setText("正在合成...")
            QApplication.processEvents()
            
            # Initialize full audio buffer if needed
            if self.synthesized_audio is None:
                # Estimate length based on original audio or F0
                # Using processor.audio length is safest if available
                if self.processor.audio is not None:
                    total_len = self.processor.audio.shape[1]
                    self.synthesized_audio = np.zeros(total_len, dtype=np.float32)
                else:
                    # Fallback
                    hop_size = self.processor.config['hop_size']
                    total_len = len(self.f0) * hop_size
                    self.synthesized_audio = np.zeros(total_len, dtype=np.float32)

            # Process segments
            hop_size = self.processor.config['hop_size']
            
            for i, (start_frame, end_frame) in enumerate(self.processor.segments):
                state = self.segment_states[i]
                
                if state['dirty'] or state['audio'] is None:
                    # Extract F0 segment
                    f0_segment = self.f0[start_frame:end_frame]
                    
                    # Synthesize segment
                    audio_seg = self.processor.synthesize_segment(i, f0_segment)
                    state['audio'] = audio_seg
                    state['dirty'] = False
                    
                    # Place into full buffer
                    start_sample = start_frame * hop_size
                    end_sample = start_sample + len(audio_seg)
                    
                    # Handle potential length mismatch at the end
                    if end_sample > len(self.synthesized_audio):
                        # Resize if needed (should be rare if initialized correctly)
                        new_len = max(len(self.synthesized_audio), end_sample)
                        new_audio = np.zeros(new_len, dtype=np.float32)
                        new_audio[:len(self.synthesized_audio)] = self.synthesized_audio
                        self.synthesized_audio = new_audio
                    
                    self.synthesized_audio[start_sample:end_sample] = audio_seg

            self.is_dirty = False
            
            # Update waveform display
            audio_np = self.synthesized_audio
            ds_factor = max(1, int(hop_size / 4))
            audio_ds = audio_np[::ds_factor]
            x_ds = np.arange(len(audio_ds)) * ds_factor / hop_size
            
            self.waveform_y = audio_ds
            self.waveform_x = x_ds
            self.update_plot()
            
            self.status_label.setText("合成完成")
        except Exception as e:
            print(f"Auto-synthesis failed: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText("自动合成失败")

    def pause_playback(self):
        if not self.is_playing: return
        
        sd.stop()
        self.is_playing = False
        self.playback_timer.stop()
        
        # Update current time one last time
        now = time.time()
        self.current_playback_time += now - self.last_wall_time
        self.status_label.setText("暂停")

    def stop_playback(self, reset=False):
        sd.stop()
        self.is_playing = False
        self.playback_timer.stop()
        
        if reset:
            self.current_playback_time = 0
            self.play_cursor.setValue(0)
            self.playback_start_time = 0
        else:
            # Return to start position
            self.current_playback_time = self.playback_start_time
            if self.processor.config and self.processor.sr:
                hop_size = self.processor.config['hop_size']
                sr = self.processor.sr
                self.play_cursor.setValue(self.current_playback_time * sr / hop_size)
                
        self.status_label.setText("停止")

    def set_playback_position(self, x_frame):
        if x_frame < 0: x_frame = 0
        
        # Update cursor visual
        self.play_cursor.setValue(x_frame)
        
        # Update internal time
        if self.processor.config and self.processor.sr:
            hop_size = self.processor.config['hop_size']
            sr = self.processor.sr
            self.current_playback_time = x_frame * hop_size / sr
            self.playback_start_time = self.current_playback_time # Update start time on seek
            
            if self.is_playing:
                # Restart playback from new position
                sd.stop()
                self.playback_timer.stop()
                self.start_playback()

    def update_cursor(self):
        if not self.is_playing: return
        
        now = time.time()
        dt = now - self.last_wall_time
        self.last_wall_time = now
        
        self.current_playback_time += dt
        
        # Convert time to x (frames)
        # x = time * sr / hop_size
        if self.processor.config:
            hop_size = self.processor.config['hop_size']
            sr = self.processor.sr
            x = self.current_playback_time * sr / hop_size
            self.play_cursor.setValue(x)
            
            # Auto scroll if cursor goes out of view?
            # view_range = self.plot_widget.viewRange()[0]
            # if x > view_range[1]:
            #     self.plot_widget.plotItem.vb.translateBy(x - view_range[0])

        # Check if finished
        if self.synthesized_audio is not None:
            if self.current_playback_time * self.processor.sr >= len(self.synthesized_audio):
                self.stop_playback()
                self.status_label.setText("播放完成")

    def update_views(self):
        self.waveform_view.setGeometry(self.plot_widget.plotItem.vb.sceneBoundingRect())
        self.waveform_view.linkedViewChanged(self.plot_widget.plotItem.vb, self.waveform_view.XAxis)

    def load_model_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, "选择模型文件夹")
        if folder:
            self.load_model(folder)

    def load_default_model(self):
        default_path = config_manager.get_default_model_path()
        if default_path and os.path.exists(default_path):
            try:
                self.processor.load_model(default_path)
                self.model_path = default_path
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"已加载默认模型: {pathlib.Path(default_path).name}")
            except Exception as e:
                if hasattr(self, 'status_label'):
                    self.status_label.setText(f"加载默认模型失败: {e}")

    def set_default_model_dialog(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择默认模型文件夹")
        if folder_path:
            config_manager.set_default_model_path(folder_path)
            QMessageBox.information(self, "设置成功", f"默认模型已设置为: {folder_path}")
            # Optionally load it now if no model is loaded
            if self.model_path is None:
                self.load_model(folder_path)

    def load_model(self, folder):
        try:
            self.status_label.setText(f"正在加载模型 {folder}...")
            QApplication.processEvents()
            
            self.processor.load_model(folder)
            self.model_path = folder
            
            self.status_label.setText(f"模型已加载: {pathlib.Path(folder).name}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            self.status_label.setText("模型加载失败。")

    def load_audio_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择音频文件", "", "音频文件 (*.wav *.flac *.mp3)")
        if file_path:
            self.load_audio(file_path)

    def load_audio(self, file_path):
        try:
            self.status_label.setText(f"正在加载音频 {file_path}...")
            QApplication.processEvents()
            
            audio_np, sr, f0_midi = self.processor.load_audio(file_path)
            self.audio_path = file_path
            
            self.f0 = f0_midi.copy()
            self.f0_original = f0_midi.copy()
            
            self.synthesized_audio = None
            self.is_dirty = True
            
            # Initialize segments
            self.segment_states = []
            for _ in self.processor.segments:
                self.segment_states.append({'dirty': True, 'audio': None})
            
            # Reset shift
            self.last_shift_value = 0.0
            self.shift_spin.blockSignals(True)
            self.shift_spin.setValue(0.0)
            self.shift_spin.blockSignals(False)
            
            # Prepare waveform for display
            hop_size = self.processor.config['hop_size']
            ds_factor = max(1, int(hop_size / 4))
            audio_ds = audio_np[::ds_factor]
            x_ds = np.arange(len(audio_ds)) * ds_factor / hop_size
            
            self.waveform_y = audio_ds
            self.waveform_x = x_ds

            self.update_plot()
            self.plot_widget.autoRange()
            self.update_views()
            self.status_label.setText("音频已加载。")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载音频失败: {str(e)}")
            self.status_label.setText("音频加载失败。")

    def export_audio_dialog(self):
        if self.synthesized_audio is None:
            QMessageBox.warning(self, "警告", "没有可导出的音频。请先合成音频。")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "导出音频", "output.wav", "WAV Audio (*.wav)")
        if file_path:
            self.export_audio(file_path)

    def export_audio(self, file_path):
        try:
            self.status_label.setText(f"正在导出到 {file_path}...")
            QApplication.processEvents()
            
            # Ensure audio is float32 and in range -1 to 1
            audio = self.synthesized_audio
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            wavfile.write(file_path, self.processor.sr, audio)
            
            self.status_label.setText(f"导出成功: {file_path}")
            QMessageBox.information(self, "成功", f"音频已导出到:\n{file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")
            self.status_label.setText("导出失败。")

    def open_project_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", "Vocal Shifter Project (*.vsp *.json)")
        if file_path:
            self.open_project(file_path)

    def open_project(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load Model
            if 'model_path' in data and os.path.exists(data['model_path']):
                self.load_model(data['model_path'])
            elif 'model_path' in data:
                QMessageBox.warning(self, "警告", f"找不到模型路径: {data['model_path']}")
            
            # Load Audio
            if 'audio_path' in data and os.path.exists(data['audio_path']):
                self.load_audio(data['audio_path'])
            elif 'audio_path' in data:
                QMessageBox.warning(self, "警告", f"找不到音频路径: {data['audio_path']}")
            
            # Restore Parameters
            if 'params' in data:
                params = data['params']
                if 'shift' in params: 
                    self.last_shift_value = params['shift']
                    self.shift_spin.setValue(params['shift'])
                if 'bpm' in params: self.bpm_spin.setValue(params['bpm'])
                if 'beats' in params: self.beats_spin.setValue(params['beats'])
            
            # Restore F0
            if 'f0' in data and self.f0 is not None:
                saved_f0 = np.array(data['f0'])
                if len(saved_f0) == len(self.f0):
                    self.f0 = saved_f0
                    self.update_plot()
                else:
                    QMessageBox.warning(self, "警告", "保存的音高数据长度与当前音频不匹配，未恢复音高编辑。")
            
            self.project_path = file_path
            self.status_label.setText(f"工程已加载: {file_path}")
            self.setWindowTitle(f"SingingVocoders - {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开工程失败: {str(e)}")

    def save_project(self):
        if self.project_path:
            self._save_project_file(self.project_path)
        else:
            self.save_project_as()

    def save_project_as(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存工程", "project.vsp", "Vocal Shifter Project (*.vsp *.json)")
        if file_path:
            self._save_project_file(file_path)

    def _save_project_file(self, file_path):
        try:
            data = {
                'version': '1.0',
                'audio_path': self.audio_path,
                'model_path': self.model_path,
                'params': {
                    'shift': self.shift_spin.value(),
                    'bpm': self.bpm_spin.value(),
                    'beats': self.beats_spin.value()
                },
                'f0': self.f0.tolist() if self.f0 is not None else None
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            
            self.project_path = file_path
            self.status_label.setText(f"工程已保存: {file_path}")
            self.setWindowTitle(f"SingingVocoders - {os.path.basename(file_path)}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存工程失败: {str(e)}")

    def update_plot(self):
        if hasattr(self, 'waveform_x') and self.waveform_x is not None:
            self.waveform_curve.setData(self.waveform_x, self.waveform_y)

        if self.f0_original is not None:
            self.f0_orig_curve_item.setData(self.f0_original, connect="finite")

        if self.f0 is not None:
            self.f0_curve_item.setData(self.f0, connect="finite")

    def on_scene_mouse_move(self, pos):
        if self.f0 is None:
            return
            
        buttons = QApplication.mouseButtons()
        is_left = bool(buttons & Qt.MouseButton.LeftButton)
        is_right = bool(buttons & Qt.MouseButton.RightButton)
        
        if not (is_left or is_right):
            self.last_mouse_pos = None
            return

        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        self.handle_draw(mouse_point, is_left, is_right)

    def on_scene_mouse_click(self, ev):
        if self.f0 is None:
            return
        
        # Check if event is already accepted (e.g. by Axis)
        if ev.isAccepted():
            return

        # Check if click is within the ViewBox geometry
        vb = self.plot_widget.plotItem.vb
        pos = ev.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
            
        if ev.button() == Qt.MouseButton.LeftButton or ev.button() == Qt.MouseButton.RightButton:
            ev.accept()
            mouse_point = vb.mapSceneToView(pos)
            
            is_left = (ev.button() == Qt.MouseButton.LeftButton)
            is_right = (ev.button() == Qt.MouseButton.RightButton)
            
            self.last_mouse_pos = None
            self.handle_draw(mouse_point, is_left, is_right)

    def handle_draw(self, point, is_left, is_right):
        x = int(point.x())
        y = point.y()
        
        # Start of a new stroke?
        if self.last_mouse_pos is None:
            self.push_undo()
        
        changed = False
        affected_range = (x, x) # Track min/max x affected
        
        if 0 <= x < len(self.f0):
            if self.last_mouse_pos is not None:
                last_x, last_y = self.last_mouse_pos
                start_x, end_x = sorted((last_x, x))
                start_x = max(0, start_x)
                end_x = min(len(self.f0) - 1, end_x)
                affected_range = (start_x, end_x)
                
                if start_x < end_x:
                    for i in range(start_x, end_x + 1):
                        if is_left:
                            ratio = (i - last_x) / (x - last_x) if x != last_x else 0
                            interp_y = last_y + ratio * (y - last_y)
                            self.f0[i] = interp_y
                            changed = True
                        elif is_right:
                            if self.f0_original is not None:
                                self.f0[i] = self.f0_original[i]
                                changed = True
                else:
                    if is_left:
                        self.f0[x] = y
                        changed = True
                    elif is_right and self.f0_original is not None:
                        self.f0[x] = self.f0_original[x]
                        changed = True
            else:
                if is_left:
                    self.f0[x] = y
                    changed = True
                elif is_right and self.f0_original is not None:
                    self.f0[x] = self.f0_original[x]
                    changed = True
            
            if changed:
                self.is_dirty = True
                # Mark affected segments as dirty
                min_x, max_x = affected_range
                for i, (seg_start, seg_end) in enumerate(self.processor.segments):
                    # Check overlap
                    if not (max_x < seg_start or min_x >= seg_end):
                        self.segment_states[i]['dirty'] = True
            
            self.last_mouse_pos = (x, y)
            self.update_plot()
            
            if changed:
                self.is_dirty = True
                self.status_label.setText("音高已修改 (未合成)")
    
    def mouseReleaseEvent(self, ev):
        self.last_mouse_pos = None
        super().mouseReleaseEvent(ev)

    def apply_shift(self, semitones):
        if self.f0 is not None:
            delta = semitones - self.last_shift_value
            self.f0 += delta
            self.last_shift_value = semitones
            self.is_dirty = True
            # Mark all segments as dirty on global shift
            for state in self.segment_states:
                state['dirty'] = True
            self.update_plot()

    def play_original(self):
        if self.processor.audio is not None:
            sd.stop()
            audio_np = self.processor.audio.numpy().T
            sd.play(audio_np, self.processor.sr)

    def synthesize_and_play(self):
        self.synthesize_audio_only()
        if self.synthesized_audio is not None:
            self.stop_playback(reset=True)
            self.start_playback()

    def stop_audio(self):
        self.stop_playback()
