import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
import pyqtgraph as pg

class CustomViewBox(pg.ViewBox):
    def __init__(self, parent_gui):
        pg.ViewBox.__init__(self)
        self.parent_gui = parent_gui

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            ev.accept()
        elif ev.button() == Qt.MouseButton.LeftButton or ev.button() == Qt.MouseButton.RightButton:
            # Ignore Left/Right so they don't trigger ViewBox default drag (Pan/Zoom)
            # This allows the Scene to handle them via sigMouseMoved for drawing
            ev.ignore()
        else:
            super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.MouseButton.MiddleButton:
            ev.accept()
            # Manual Pan
            # mapToView maps from Item coordinates (pixels) to Data coordinates
            p1 = self.mapToView(ev.pos())
            p2 = self.mapToView(ev.lastPos())
            tr = p1 - p2
            self.translateBy(-tr)
        else:
            super().mouseMoveEvent(ev)

    def wheelEvent(self, ev, axis=None):
        modifiers = QApplication.keyboardModifiers()
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            # Zoom X (axis 0)
            super().wheelEvent(ev, axis=0)
        elif modifiers == Qt.KeyboardModifier.AltModifier:
            # Zoom Y (axis 1)
            super().wheelEvent(ev, axis=1)
        else:
            super().wheelEvent(ev, axis=axis)
            
    def mouseClickEvent(self, ev):
        # Ignore Left/Right clicks so they propagate or don't trigger ViewBox menus
        if ev.button() == Qt.MouseButton.LeftButton or ev.button() == Qt.MouseButton.RightButton:
            ev.ignore()
        else:
            super().mouseClickEvent(ev)

class PianoRollAxis(pg.AxisItem):
    def __init__(self, orientation='left', **kwargs):
        super().__init__(orientation, **kwargs)

    def tickStrings(self, values, scale, spacing):
        strings = []
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for v in values:
            # Only label if close to integer
            if abs(v - round(v)) > 0.1:
                strings.append("")
                continue
            
            try:
                # v is MIDI note number
                note_idx = int(round(v))
                octave = note_idx // 12 - 1
                name = note_names[note_idx % 12]
                strings.append(f"{name}{octave}")
            except:
                strings.append("")
        return strings
    
    def tickValues(self, minVal, maxVal, size):
        # Force integer ticks
        min_idx = int(np.ceil(minVal))
        max_idx = int(np.floor(maxVal))
        values = list(range(min_idx, max_idx + 1))
        return [(1.0, values)]

class BPMAxis(pg.AxisItem):
    def __init__(self, parent_gui, orientation='top', **kwargs):
        super().__init__(orientation, **kwargs)
        self.parent_gui = parent_gui

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            ev.accept()
            vb = self.linkedView()
            if vb:
                pos = ev.scenePos()
                view_point = vb.mapSceneToView(pos)
                x = view_point.x()
                self.parent_gui.set_playback_position(x)

    def tickStrings(self, values, scale, spacing):
        # Access config and sr via processor
        if not hasattr(self.parent_gui, 'processor') or \
           not self.parent_gui.processor.config or \
           not self.parent_gui.processor.sr:
            return []
            
        hop_size = self.parent_gui.processor.config.get('hop_size', 512)
        sr = self.parent_gui.processor.config.get('audio_sample_rate', 44100)
        bpm = self.parent_gui.bpm_spin.value()
        beats_per_bar = self.parent_gui.beats_spin.value()
        
        strings = []
        for v in values:
            # v is frame index
            time_sec = v * hop_size / sr
            total_beats = time_sec * (bpm / 60.0)
            bar = int(total_beats / beats_per_bar) + 1
            beat = int(total_beats % beats_per_bar) + 1
            strings.append(f"{bar}-{beat}")
        return strings
