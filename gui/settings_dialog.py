"""
PyQt dialog for setting final analysis parameters.
Updated to ensure all TST spinboxes trigger video preview updates.
"""
import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                             QSlider, QVBoxLayout, QGroupBox, QWidget, QScrollArea)

from core.video import VideoHandler
from mazes.tst import TST

class SettingsDialog(QDialog):
    """A comprehensive dialog for setting analysis parameters."""
    
    def __init__(self, video_path: str, settings, is_verification=False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Settings")
        self.resize(700, 600)
        
        self.video_info = VideoHandler(video_path)
        self.cap = cv2.VideoCapture(str(video_path))
        self.settings = settings
        self.result_action = "cancel"
        
        # Detect TST mode
        self.is_tst = any(k.startswith('mouse_') for k in self.settings.roi_manager.rois.keys())
        
        self.mouse_time_widgets = {} # {name: (start_spin, end_spin)}
        
        self.init_ui(is_verification)
        self.update_preview(int(self.settings.start_time))

    def init_ui(self, is_verification: bool):
        main_layout = QVBoxLayout()

        # 1. Video Preview
        self.preview_label = QLabel()
        self.preview_label.setFixedSize(640, 360)
        self.preview_label.setStyleSheet("background-color: black; border: 1px solid grey;")
        main_layout.addWidget(self.preview_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 2. Time Configuration (Dynamic)
        time_group = QGroupBox("Time Range Configuration")
        time_layout = QVBoxLayout()
        
        max_duration = self.video_info.duration
        
        if self.is_tst:
            # --- TST MULTI-MOUSE MODE ---
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFixedHeight(150)
            
            container = QWidget()
            form = QFormLayout(container)
            
            # Get mouse names from ROIs
            mice = sorted([k for k in self.settings.roi_manager.rois.keys() if k != 'outside'])
            
            for mouse in mice:
                row_layout = QHBoxLayout()
                
                # Defaults
                s_val = 0.0
                e_val = max_duration
                
                if self.settings.per_roi_times and mouse in self.settings.per_roi_times:
                    s_val, e_val = self.settings.per_roi_times[mouse]
                
                start_spin = QDoubleSpinBox()
                start_spin.setRange(0, max_duration)
                start_spin.setValue(s_val)
                start_spin.setSuffix("s")
                
                end_spin = QDoubleSpinBox()
                end_spin.setRange(0, max_duration)
                end_spin.setValue(e_val)
                end_spin.setSuffix("s")
                
                row_layout.addWidget(QLabel("Start:"))
                row_layout.addWidget(start_spin)
                row_layout.addWidget(QLabel("End:"))
                row_layout.addWidget(end_spin)
                
                # --- FIX: Connect EVERY spinbox to update_preview ---
                # Use a closure (lambda v=None: ...) to safely capture loop variables if needed,
                # though sender() access inside update_preview isn't needed here, just the value.
                # We pass the value directly.
                start_spin.valueChanged.connect(lambda v: self.update_preview(v))
                end_spin.valueChanged.connect(lambda v: self.update_preview(v))
                
                self.mouse_time_widgets[mouse] = (start_spin, end_spin)
                form.addRow(f"{mouse.replace('_',' ').title()}:", row_layout)
            
            scroll.setWidget(container)
            time_layout.addWidget(QLabel("Set individual start/end times for each mouse:"))
            time_layout.addWidget(scroll)
            
        else:
            # --- STANDARD SINGLE MODE ---
            slider_max = int(max_duration)

            self.start_slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=slider_max, value=int(self.settings.start_time))
            self.start_spinbox = QDoubleSpinBox(decimals=2, minimum=0, maximum=max_duration, value=self.settings.start_time)
            start_hbox = QHBoxLayout(); start_hbox.addWidget(self.start_slider); start_hbox.addWidget(self.start_spinbox)

            end_val = self.settings.end_time or max_duration
            self.end_slider = QSlider(Qt.Orientation.Horizontal, minimum=0, maximum=slider_max, value=int(end_val))
            self.end_spinbox = QDoubleSpinBox(decimals=2, minimum=0, maximum=max_duration, value=end_val)
            end_hbox = QHBoxLayout(); end_hbox.addWidget(self.end_slider); end_hbox.addWidget(self.end_spinbox)

            time_layout.addLayout(start_hbox)
            time_layout.addLayout(end_hbox)
            
            self.start_slider.valueChanged.connect(self.on_slider_change)
            self.end_slider.valueChanged.connect(self.on_slider_change)
            self.start_spinbox.valueChanged.connect(self.on_spinbox_change)
            self.end_spinbox.valueChanged.connect(self.on_spinbox_change)

        time_group.setLayout(time_layout)
        main_layout.addWidget(time_group)

        # 3. Other Settings
        other_layout = QFormLayout()
        
        if not self.is_tst:
            ref_len = self.settings.roi_manager.reference_length_pixels
            ref_name = self.settings.roi_manager.reference_name
            ref_label = QLabel(f"{ref_name} ({ref_len:.2f} px)")
            
            self.scale_spin = QDoubleSpinBox(decimals=4, suffix=" cm", minimum=0, maximum=1000)
            if self.settings.scale_factor > 0 and ref_len > 0:
                self.scale_spin.setValue(self.settings.scale_factor * ref_len)
            
            other_layout.addRow("Reference:", ref_label)
            other_layout.addRow("Real-world length:", self.scale_spin)
        
        # --- IMPROVEMENT: TST doesn't need timelapse, it needs validation video ---
        check_label = "Create Validation Video (Annotated)" if self.is_tst else "Create Timelapse Video"
        self.timelapse_check = QCheckBox(check_label)
        self.timelapse_check.setChecked(self.settings.create_timelapse)
        other_layout.addRow(self.timelapse_check)
        
        main_layout.addLayout(other_layout)

        # 4. Buttons
        self.button_box = QDialogButtonBox()
        if is_verification:
            self.accept_btn = self.button_box.addButton("Accept", QDialogButtonBox.ButtonRole.AcceptRole)
            self.redraw_btn = self.button_box.addButton("Redraw ROIs", QDialogButtonBox.ButtonRole.ActionRole)
            self.skip_btn = self.button_box.addButton("Skip Video", QDialogButtonBox.ButtonRole.DestructiveRole)
        else:
            self.accept_btn = self.button_box.addButton("OK", QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_btn = self.button_box.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        main_layout.addWidget(self.button_box)

        self.setLayout(main_layout)
        
        self.accept_btn.clicked.connect(self.on_accept)
        self.cancel_btn.clicked.connect(self.reject)
        if hasattr(self, 'redraw_btn'):
            self.redraw_btn.clicked.connect(self.on_redraw)
            self.skip_btn.clicked.connect(self.on_skip)

    def on_slider_change(self):
        self._sync_widgets(source='slider')
        self.update_preview(self.sender().value())

    def on_spinbox_change(self):
        self._sync_widgets(source='spinbox')
        self.update_preview(int(self.sender().value()))

    def _sync_widgets(self, source: str):
        for w in [self.start_slider, self.end_slider, self.start_spinbox, self.end_spinbox]:
            w.blockSignals(True)
        if source == 'slider':
            if self.start_slider.value() > self.end_slider.value():
                self.start_slider.setValue(self.end_slider.value())
            self.start_spinbox.setValue(self.start_slider.value())
            self.end_spinbox.setValue(self.end_slider.value())
        elif source == 'spinbox':
            if self.start_spinbox.value() > self.end_spinbox.value():
                self.start_spinbox.setValue(self.end_spinbox.value())
            self.start_slider.setValue(int(self.start_spinbox.value()))
            self.end_slider.setValue(int(self.end_spinbox.value()))
        for w in [self.start_slider, self.end_slider, self.start_spinbox, self.end_spinbox]:
            w.blockSignals(False)

    def on_accept(self): 
        self.result_action = "accept"
        self.accept()
    def on_redraw(self): 
        self.result_action = "redraw"
        self.accept()
    def on_skip(self): 
        self.result_action = "skip"
        self.accept()

    def update_preview(self, time_in_seconds: float):
        frame_idx = int(time_in_seconds * self.video_info.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if ret:
            vis_frame = self.settings.roi_manager.draw_on_frame(frame)
            img = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            q_img = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.preview_label.setPixmap(pixmap.scaled(
                self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def get_settings(self):
        if self.is_tst:
            per_roi_times = {}
            global_start = float('inf')
            global_end = 0.0
            
            for name, (s_spin, e_spin) in self.mouse_time_widgets.items():
                s = s_spin.value()
                e = e_spin.value()
                per_roi_times[name] = (s, e)
                global_start = min(global_start, s)
                global_end = max(global_end, e)
            
            self.settings.per_roi_times = per_roi_times
            self.settings.start_time = global_start
            self.settings.end_time = global_end
            self.settings.scale_factor = 0.0
            
        else:
            end_time = self.end_spinbox.value()
            if end_time >= self.video_info.duration - 0.1: end_time = None
            
            self.settings.start_time = self.start_spinbox.value()
            self.settings.end_time = end_time
            
            scale_val = self.scale_spin.value()
            ref_len = self.settings.roi_manager.reference_length_pixels
            scale_factor = (scale_val / ref_len) if scale_val > 0 and ref_len > 0 else 0.0
            self.settings.scale_factor = scale_factor

        self.settings.create_timelapse = self.timelapse_check.isChecked()
        return self.settings

    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)