#!/usr/bin/env python3
"""
goal_gui_node — 桌面 GUI 点位导航工具

基于 PyQt5，显示 Nav2 地图 (2026rc.pgm)，在地图上点击点位并发送导航目标。
"""

import math
import os
import sys
import threading
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor

from nav2_msgs.action import NavigateToPose
from nav2_msgs.action import NavigateThroughPoses
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_ros import TransformException
import yaml

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsTextItem,
    QGraphicsPolygonItem, QGraphicsDropShadowEffect,
    QVBoxLayout, QHBoxLayout, QSplitter, QScrollArea,
    QPushButton, QLabel, QGroupBox, QFormLayout, QListWidget,
    QListWidgetItem, QMessageBox, QInputDialog, QDialog,
    QDialogButtonBox, QLineEdit, QDoubleSpinBox, QTextEdit,
    QStatusBar, QCheckBox, QSizePolicy, QFrame,
)
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont,
    QTransform, QPolygonF, QCursor, QIcon,
)
from PyQt5.QtCore import (
    Qt, QPointF, QRectF, QTimer, pyqtSignal, pyqtSlot, QObject,
)


# =========================================================================
# 辅助函数（复刻 goal_console_node.py）
# =========================================================================

def yaw_to_quaternion(yaw: float):
    return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def quaternion_to_yaw(qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz)
    cosy = 1.0 - 2.0 * (qz * qz)
    return math.atan2(siny, cosy)


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


# =========================================================================
# 地图坐标转换
# =========================================================================

class MapTransform:
    """PGM 像素 ↔ ROS map 坐标转换"""

    def __init__(self, resolution: float, origin_x: float, origin_y: float,
                 image_w: int, image_h: int):
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.image_w = image_w
        self.image_h = image_h

    def map_to_pixel(self, mx: float, my: float) -> tuple:
        """map 坐标 → 原始 PGM 像素"""
        px = (mx - self.origin_x) / self.resolution
        py = self.image_h - (my - self.origin_y) / self.resolution
        return (px, py)

    def pixel_to_map(self, px: float, py: float) -> tuple:
        """原始 PGM 像素 → map 坐标"""
        mx = self.origin_x + px * self.resolution
        my = self.origin_y + (self.image_h - py) * self.resolution
        return (mx, my)


def occupancy_to_qimage(msg: OccupancyGrid, alpha_base: int = 80) -> QImage:
    """将 OccupancyGrid 转为带透明度的 QImage"""
    w = msg.info.width
    h = msg.info.height
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))  # 全透明

    data = msg.data
    for row in range(h):
        # OccupancyGrid data 从左上角开始（row 0 = 顶部），需要翻转到 QImage 的底部
        src_row = h - 1 - row
        for col in range(w):
            idx = src_row * w + col
            val = data[idx]
            a = 0
            r, g, b = 0, 0, 0
            if val == -1:  # unknown
                a = 20
                r, g, b = 128, 128, 128
            elif val == 0:  # free
                a = 0
            elif 1 <= val <= 30:
                a = alpha_base
                r, g, b = 0, 255, 0
            elif 31 <= val <= 60:
                a = alpha_base + 30
                r, g, b = 255, 255, 0
            elif 61 <= val <= 99:
                a = alpha_base + 60
                r, g, b = 255, 128, 0
            else:  # 100 = lethal
                a = alpha_base + 100
                r, g, b = 255, 0, 0

            if a > 255:
                a = 255
            img.setPixelColor(col, row, QColor(r, g, b, a))
    return img


# =========================================================================
# 机器人箭头
# =========================================================================

def make_arrow_polygon(size: float = 12.0) -> QPolygonF:
    """绘制朝东(0°)的三角箭头，后续用 rotate 旋转到实际 yaw"""
    half = size / 2.0
    poly = QPolygonF([
        QPointF(size, 0),          # 尖端
        QPointF(-half, -half * 0.7),  # 左上
        QPointF(0, 0),             # 中心
        QPointF(-half, half * 0.7),   # 左下
    ])
    return poly


# =========================================================================
# 地图 Widget（QGraphicsView）
# =========================================================================

class MapScene(QGraphicsScene):
    """地图场景，管理所有图元"""
    point_clicked_signal = pyqtSignal(float, float)  # map_x, map_y

    def __init__(self, parent=None):
        super().__init__(parent)
        self._point_items: list[QGraphicsEllipseItem] = []
        self._text_items: list[QGraphicsTextItem] = []
        self._robot_item: Optional[QGraphicsPolygonItem] = None
        self._click_marker: Optional[QGraphicsEllipseItem] = None
        # 路线标注
        self._route_labels: list[QGraphicsTextItem] = []
        self._route_lines: list = []  # QGraphicsLineItem
        self._route_current_highlight: Optional[QGraphicsEllipseItem] = None
        # costmap overlay
        self._global_costmap_item: Optional[QGraphicsPixmapItem] = None
        self._local_costmap_item: Optional[QGraphicsPixmapItem] = None

    def add_point(self, px: float, py: float, name: str, selected: bool = False):
        """添加点位标记"""
        r = 4.0
        color = QColor(255, 80, 80) if selected else QColor(255, 165, 0)
        ellipse = QGraphicsEllipseItem(px - r, py - r, r * 2, r * 2)
        ellipse.setPen(QPen(color, 1.5))
        ellipse.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 100)))
        self.addItem(ellipse)
        self._point_items.append(ellipse)

        text = QGraphicsTextItem(name)
        text.setDefaultTextColor(QColor(255, 255, 255))
        text.setFont(QFont('Sans', 8, QFont.Bold))
        text.setPos(px + 6, py - 14)
        self.addItem(text)
        self._text_items.append(text)

    def remove_all_points(self):
        for item in self._point_items:
            self.removeItem(item)
        for item in self._text_items:
            self.removeItem(item)
        self._point_items.clear()
        self._text_items.clear()
        self._remove_all_route_markers()

    def _remove_all_route_markers(self):
        for item in self._route_labels:
            self.removeItem(item)
        for item in self._route_lines:
            self.removeItem(item)
        self._route_labels.clear()
        self._route_lines.clear()
        if self._route_current_highlight:
            self.removeItem(self._route_current_highlight)
            self._route_current_highlight = None

    def set_global_costmap_item(self, item):
        if self._global_costmap_item:
            self.removeItem(self._global_costmap_item)
            self._global_costmap_item = None
        if item:
            self._global_costmap_item = item
            self.addItem(item)

    def set_local_costmap_item(self, item):
        if self._local_costmap_item:
            self.removeItem(self._local_costmap_item)
            self._local_costmap_item = None
        if item:
            self._local_costmap_item = item
            self.addItem(item)

    def draw_route_markers(self, route_point_pixels: list, current_index: int = -1):
        """绘制路线序号标签和连接线。
        route_point_pixels: [(px, py, name), ...]
        current_index: 当前执行点索引 (-1 表示无)"""
        self._remove_all_route_markers()

        from PyQt5.QtWidgets import QGraphicsLineItem

        # 连接线（灰色虚线）
        for i in range(len(route_point_pixels) - 1):
            x1, y1, _ = route_point_pixels[i]
            x2, y2, _ = route_point_pixels[i + 1]
            line = QGraphicsLineItem(x1, y1, x2, y2)
            line.setPen(QPen(QColor(100, 200, 255, 180), 2, Qt.DashLine))
            self.addItem(line)
            self._route_lines.append(line)

        # 序号标签（带圆圈背景）
        from PyQt5.QtWidgets import QGraphicsEllipseItem
        for i, (px, py, _) in enumerate(route_point_pixels):
            # 小圆背景
            r = 5
            bg = QColor(30, 144, 255, 220) if i == current_index else QColor(30, 144, 255, 140)
            circle = QGraphicsEllipseItem(px - r, py - r, r * 2, r * 2)
            circle.setPen(QPen(QColor(30, 144, 255), 1.5))
            circle.setBrush(QBrush(bg))
            self.addItem(circle)
            self._route_labels.append(circle)

            # 序号文字
            label = QGraphicsTextItem(str(i + 1))
            label.setDefaultTextColor(QColor(255, 255, 255))
            font = QFont('Sans', 8, QFont.Bold)
            label.setFont(font)
            rect = label.boundingRect()
            label.setPos(px - rect.width() / 2, py - rect.height() / 2)
            self.addItem(label)
            self._route_labels.append(label)

        # 当前执行点高亮
        if 0 <= current_index < len(route_point_pixels):
            px, py, _ = route_point_pixels[current_index]
            r = 7
            hl = QGraphicsEllipseItem(px - r, py - r, r * 2, r * 2)
            hl.setPen(QPen(QColor(0, 255, 100), 2))
            hl.setBrush(QBrush(QColor(0, 255, 100, 60)))
            self.addItem(hl)
            self._route_current_highlight = hl

    def set_robot_pose(self, px: float, py: float, yaw_rad: float):
        """更新机器人箭头"""
        if self._robot_item:
            self.removeItem(self._robot_item)

        arrow = make_arrow_polygon(14.0)
        # yaw 从"朝东=0°"开始，Qt 坐标系 y 向下，需要翻转
        # ROS yaw: 0=东, 90=北 (逆时针)
        # 映射到屏幕：Qt 的 0°=右(东)，顺时针为正
        transform = QTransform()
        transform.translate(px, py)
        transform.rotate(-rad_to_deg(yaw_rad))  # Qt 顺时针为正
        poly = transform.map(arrow)

        self._robot_item = QGraphicsPolygonItem(poly)
        self._robot_item.setPen(QPen(QColor(0, 180, 0), 2))
        self._robot_item.setBrush(QBrush(QColor(0, 220, 0, 180)))
        self.addItem(self._robot_item)

    def set_click_marker(self, px: float, py: float):
        """设置点击位置标记"""
        if self._click_marker:
            self.removeItem(self._click_marker)
        r = 3
        self._click_marker = QGraphicsEllipseItem(px - r, py - r, r * 2, r * 2)
        self._click_marker.setPen(QPen(QColor(0, 200, 255), 2))
        self._click_marker.setBrush(QBrush(QColor(0, 200, 255, 120)))
        self.addItem(self._click_marker)

    def mousePressEvent(self, event):
        """捕获点击事件，发送 map 坐标信号"""
        pos = event.scenePos()
        # 这里只是记录，实际坐标转换在 MapWidget 中处理
        # 由 MapWidget 连接信号
        super().mousePressEvent(event)


class MapWidget(QGraphicsView):
    """地图显示控件，支持缩放和拖拽"""
    mouse_moved_signal = pyqtSignal(float, float)   # map_x, map_y
    map_clicked_signal = pyqtSignal(float, float)    # map_x, map_y

    def __init__(self, map_transform: MapTransform, pgm_path: str, parent=None):
        super().__init__(parent)
        self._map_tf = map_transform
        self._scene = MapScene(self)
        self.setScene(self._scene)

        # 性能优化：低开销绘制
        self.setRenderHints(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.BoundingRectViewportUpdate)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setCacheMode(QGraphicsView.CacheBackground)

        # 拖拽平移
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # 缩放
        self._zoom_level = 0
        self._zoom_factor = 1.15
        self._min_zoom = -20
        self._max_zoom = 20

        # 加载地图图片
        self._raw_pixmap = QPixmap(pgm_path)
        if self._raw_pixmap.isNull():
            print(f'[ERROR] 无法加载地图: {pgm_path}')
            self._map_item = None
        else:
            self._map_item = QGraphicsPixmapItem(self._raw_pixmap)
            self._map_item.setTransformationMode(Qt.SmoothTransformation)
            self._scene.addItem(self._map_item)
            self._scene.setSceneRect(QRectF(self._raw_pixmap.rect()))

        # view 层旋转
        self._display_rotation = 270
        self._apply_view_transform()

        # 鼠标追踪
        self.setMouseTracking(True)

        # 静态变量：当前悬停坐标
        self._hover_map_xy = (0.0, 0.0)

    @property
    def scene_ref(self) -> MapScene:
        return self._scene

    def wheelEvent(self, event):
        """滚轮缩放"""
        delta = event.angleDelta().y()
        if delta > 0 and self._zoom_level < self._max_zoom:
            self.scale(self._zoom_factor, self._zoom_factor)
            self._zoom_level += 1
        elif delta < 0 and self._zoom_level > self._min_zoom:
            self.scale(1.0 / self._zoom_factor, 1.0 / self._zoom_factor)
            self._zoom_level -= 1

    def keyPressEvent(self, event):
        """R 键重置视图"""
        if event.key() == Qt.Key_R:
            self.reset_view()
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动：计算并发送 map 坐标"""
        pos = self.mapToScene(event.pos())
        px, py = pos.x(), pos.y()
        if 0 <= px <= self._map_tf.image_w and 0 <= py <= self._map_tf.image_h:
            mx, my = self._map_tf.pixel_to_map(px, py)
            self._hover_map_xy = (mx, my)
            self.mouse_moved_signal.emit(mx, my)
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        """鼠标点击：发送 map 坐标"""
        pos = self.mapToScene(event.pos())
        px, py = pos.x(), pos.y()
        if 0 <= px <= self._map_tf.image_w and 0 <= py <= self._map_tf.image_h:
            mx, my = self._map_tf.pixel_to_map(px, py)
            self.map_clicked_signal.emit(mx, my)
            self._scene.set_click_marker(px, py)
        super().mousePressEvent(event)

    def reset_view(self):
        self._display_rotation = 270
        self._zoom_level = 0
        self.resetTransform()
        self._apply_view_transform()

    def _apply_view_transform(self):
        t = QTransform().rotate(self._display_rotation)
        t.scale(self._zoom_factor ** self._zoom_level, self._zoom_factor ** self._zoom_level)
        self.setTransform(t)
        # 调试：验证costmap是否在同一scene
        gcm = self._scene._global_costmap_item
        if gcm:
            print(f'[GCM] item_in_scene={gcm.scene() is self._scene}, rotation={self._display_rotation}, pos={gcm.pos()}, z={gcm.zValue()}')

    def set_display_rotation(self, deg: int):
        self._display_rotation = deg
        self._apply_view_transform()

    def wheelEvent(self, event):
        """滚轮缩放"""
        delta = event.angleDelta().y()
        if delta > 0 and self._zoom_level < self._max_zoom:
            self._zoom_level += 1
        elif delta < 0 and self._zoom_level > self._min_zoom:
            self._zoom_level -= 1
        self._apply_view_transform()


# =========================================================================
# 点位编辑对话框
# =========================================================================

class PointEditDialog(QDialog):
    """点位编辑/添加对话框"""

    def __init__(self, title: str, point: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(350)

        layout = QFormLayout(self)

        self.name_edit = QLineEdit()
        self.x_spin = QDoubleSpinBox()
        self.y_spin = QDoubleSpinBox()
        self.yaw_spin = QDoubleSpinBox()
        self.desc_edit = QTextEdit()

        for sp in [self.x_spin, self.y_spin]:
            sp.setRange(-999, 999)
            sp.setDecimals(4)
        self.yaw_spin.setRange(-180, 180)
        self.yaw_spin.setDecimals(2)
        self.yaw_spin.setSuffix('°')
        self.desc_edit.setMaximumHeight(60)

        if point:
            self.name_edit.setText(point.get('name', ''))
            self.x_spin.setValue(point.get('x', 0))
            self.y_spin.setValue(point.get('y', 0))
            self.yaw_spin.setValue(point.get('yaw_deg', 0))
            self.desc_edit.setText(point.get('description', ''))

        layout.addRow('名称:', self.name_edit)
        layout.addRow('X (map):', self.x_spin)
        layout.addRow('Y (map):', self.y_spin)
        layout.addRow('角度:', self.yaw_spin)
        layout.addRow('描述:', self.desc_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_point(self) -> dict:
        yaw_deg = self.yaw_spin.value()
        return {
            'name': self.name_edit.text().strip(),
            'x': self.x_spin.value(),
            'y': self.y_spin.value(),
            'yaw': deg_to_rad(yaw_deg),
            'yaw_deg': yaw_deg,
            'description': self.desc_edit.toPlainText().strip(),
        }


# =========================================================================
# 控制面板（右侧）
# =========================================================================

class ControlPanel(QTabWidget):
    """右侧控制面板（分页）"""

    # 信号（由 GoalGuiNode 连接）
    refresh_points = pyqtSignal()
    save_points = pyqtSignal()
    add_clicked = pyqtSignal()
    add_robot_pose = pyqtSignal()
    edit_point = pyqtSignal(int)
    delete_point = pyqtSignal(int)
    send_goal = pyqtSignal(int, bool)  # index, skip_confirm
    cancel_goal = pyqtSignal()
    refresh_robot = pyqtSignal()
    # 路线信号
    add_to_route = pyqtSignal(int)       # point_index
    remove_from_route = pyqtSignal(int)  # route_index
    move_route_up = pyqtSignal(int)
    move_route_down = pyqtSignal(int)
    clear_route = pyqtSignal()
    start_route = pyqtSignal()
    stop_route = pyqtSignal()
    # 图层切换信号
    layer_changed = pyqtSignal()
    # 连续添加模式
    continuous_add_toggled = pyqtSignal(bool)
    # 路线库信号
    refresh_routes_lib = pyqtSignal()
    preview_route = pyqtSignal(int)
    load_route_to_queue = pyqtSignal(int)
    start_lib_route = pyqtSignal(int)
    save_current_as_route = pyqtSignal()
    delete_route = pyqtSignal(int)
    # 阵营信号
    team_changed = pyqtSignal(str)
    record_start = pyqtSignal(str)
    # 地图方向
    map_rotation = pyqtSignal(int)
    map_flip_x = pyqtSignal()
    map_flip_y = pyqtSignal()
    map_reset_dir = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_index: int = -1
        self._init_ui()

    def _init_ui(self):
        # Tab 1: 点位
        points_tab = QWidget()
        pl = QVBoxLayout(points_tab)
        list_group = QGroupBox('预存点位')
        ll = QVBoxLayout(list_group)
        self._point_list = QListWidget()
        self._point_list.setMinimumHeight(100)
        self._point_list.setMaximumHeight(200)
        self._point_list.currentRowChanged.connect(self._on_selection_changed)
        ll.addWidget(self._point_list)
        pl.addWidget(list_group)

        info_group = QGroupBox('选中点位')
        il = QFormLayout(info_group)
        self._lbl_name = QLabel('-'); self._lbl_xy = QLabel('-')
        self._lbl_yaw = QLabel('-'); self._lbl_desc = QLabel('-')
        self._lbl_desc.setWordWrap(True)
        il.addRow('名称:', self._lbl_name); il.addRow('坐标:', self._lbl_xy)
        il.addRow('角度:', self._lbl_yaw); il.addRow('描述:', self._lbl_desc)
        pl.addWidget(info_group)

        btn_refresh_pts = QPushButton('刷新点位'); btn_save = QPushButton('保存点位')
        btn_add_clicked = QPushButton('添加点击点'); btn_add_robot = QPushButton('添加机器人位置')
        btn_edit = QPushButton('编辑选中点'); btn_delete = QPushButton('删除选中点')
        r1, r2, r3 = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        r1.addWidget(btn_refresh_pts); r1.addWidget(btn_save)
        r2.addWidget(btn_add_clicked); r2.addWidget(btn_add_robot)
        r3.addWidget(btn_edit); r3.addWidget(btn_delete)
        pl.addLayout(r1); pl.addLayout(r2); pl.addLayout(r3)

        # 阵营选择
        team_group = QGroupBox('阵营/起始点')
        tl = QVBoxLayout(team_group)
        tr1 = QHBoxLayout()
        self._btn_team_red = QPushButton('红方')
        self._btn_team_red.setStyleSheet('QPushButton { background-color: #e74c3c; color: white; font-weight: bold; }')
        self._btn_team_blue = QPushButton('蓝方')
        self._btn_team_blue.setStyleSheet('QPushButton { background-color: #3498db; color: white; }')
        tr1.addWidget(self._btn_team_red); tr1.addWidget(self._btn_team_blue)
        tl.addLayout(tr1)
        self._lbl_team_info = QLabel('当前: 红方  |  起点: red_start')
        self._lbl_team_info.setWordWrap(True)
        tl.addWidget(self._lbl_team_info)
        tr2 = QHBoxLayout()
        self._btn_record_red_start = QPushButton('记录当前为红方起点')
        self._btn_record_blue_start = QPushButton('记录当前为蓝方起点')
        tr2.addWidget(self._btn_record_red_start); tr2.addWidget(self._btn_record_blue_start)
        tl.addLayout(tr2)
        hint = QLabel('切换阵营只改变 GUI 默认起点，不会重置定位。\n若 map→base_link 不正确，需重新对齐定位。')
        hint.setWordWrap(True); hint.setStyleSheet('color: gray; font-size: 10px;')
        tl.addWidget(hint)
        pl.addWidget(team_group)
        pl.addStretch()
        self.addTab(points_tab, '点位')

        # Tab 2: 路线
        route_tab = QWidget()
        rl = QVBoxLayout(route_tab)
        self._route_list = QListWidget()
        self._route_list.setMinimumHeight(80); self._route_list.setMaximumHeight(160)
        rsl = QHBoxLayout()
        self._lbl_route_count = QLabel('0 个点'); self._lbl_route_state = QLabel('idle')
        self._lbl_route_state.setStyleSheet('font-weight: bold;')
        rsl.addWidget(self._lbl_route_count); rsl.addStretch()
        rsl.addWidget(QLabel('状态:')); rsl.addWidget(self._lbl_route_state)
        rl.addLayout(rsl); rl.addWidget(self._route_list)
        self._btn_add_route = QPushButton('加入路线'); self._btn_remove_route = QPushButton('移除')
        self._btn_move_up = QPushButton('上移'); self._btn_move_down = QPushButton('下移')
        self._btn_clear_route = QPushButton('清空路线')
        self._btn_start_route = QPushButton('开始路线')
        self._btn_start_route.setStyleSheet('QPushButton { background-color: #2196F3; color: white; font-weight: bold; }')
        self._btn_stop_route = QPushButton('停止路线')
        self._btn_stop_route.setStyleSheet('QPushButton { background-color: #f44336; color: white; font-weight: bold; }')
        rr1, rr2, rr3 = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        rr1.addWidget(self._btn_add_route); rr1.addWidget(self._btn_remove_route)
        rr2.addWidget(self._btn_move_up); rr2.addWidget(self._btn_move_down); rr2.addWidget(self._btn_clear_route)
        rr3.addWidget(self._btn_start_route); rr3.addWidget(self._btn_stop_route)
        rl.addLayout(rr1); rl.addLayout(rr2); rl.addLayout(rr3)
        rl.addStretch()
        self.addTab(route_tab, '路线')

        # Tab 3: 路线库
        lib_tab = QWidget()
        lib_l = QVBoxLayout(lib_tab)
        self._route_lib_list = QListWidget()
        self._route_lib_list.setMinimumHeight(80); self._route_lib_list.setMaximumHeight(150)
        lib_l.addWidget(QLabel('预设路线:'))
        lib_l.addWidget(self._route_lib_list)
        self._lbl_route_lib_info = QLabel('')
        self._lbl_route_lib_info.setWordWrap(True)
        lib_l.addWidget(self._lbl_route_lib_info)
        lr1, lr2, lr3 = QHBoxLayout(), QHBoxLayout(), QHBoxLayout()
        btn_refresh_lib = QPushButton('刷新路线库')
        btn_preview_lib = QPushButton('预览路线')
        lr1.addWidget(btn_refresh_lib); lr1.addWidget(btn_preview_lib)
        btn_load_lib = QPushButton('加载到路线队列')
        btn_start_lib = QPushButton('开始该路线')
        btn_start_lib.setStyleSheet('QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }')
        lr2.addWidget(btn_load_lib); lr2.addWidget(btn_start_lib)
        btn_save_lib = QPushButton('保存当前路线为模板')
        btn_delete_lib = QPushButton('删除路线')
        lr3.addWidget(btn_save_lib); lr3.addWidget(btn_delete_lib)
        lib_l.addLayout(lr1); lib_l.addLayout(lr2); lib_l.addLayout(lr3)
        lib_l.addStretch()
        self.addTab(lib_tab, '路线库')

        # Tab 4: 图层
        layer_tab = QWidget()
        ltl = QVBoxLayout(layer_tab)
        self._cb_static_map = QCheckBox('静态地图')
        self._cb_global_cm = QCheckBox('Global Costmap')
        self._cb_local_cm = QCheckBox('Local Costmap')
        self._cb_points = QCheckBox('点位')
        self._cb_routes = QCheckBox('路线')
        self._cb_robot = QCheckBox('机器人')
        for cb in [self._cb_static_map, self._cb_global_cm, self._cb_local_cm,
                   self._cb_points, self._cb_routes, self._cb_robot]:
            cb.setChecked(True); cb.toggled.connect(self.layer_changed.emit)
        ltl.addWidget(self._cb_static_map); ltl.addWidget(self._cb_global_cm)
        ltl.addWidget(self._cb_local_cm); ltl.addWidget(self._cb_points)
        ltl.addWidget(self._cb_routes); ltl.addWidget(self._cb_robot)
        self._cb_continuous = QCheckBox('连续添加模式')
        self._cb_continuous.setChecked(False)
        self._cb_continuous.toggled.connect(self.continuous_add_toggled.emit)
        ltl.addWidget(self._cb_continuous)
        ltl.addStretch()

        # 地图方向
        dir_group = QGroupBox('地图方向')
        dl = QVBoxLayout(dir_group)
        dr1, dr2 = QHBoxLayout(), QHBoxLayout()
        self._btn_rot0 = QPushButton('0°'); self._btn_rot90 = QPushButton('90°')
        self._btn_rot180 = QPushButton('180°'); self._btn_rot270 = QPushButton('270°')
        dr1.addWidget(self._btn_rot0); dr1.addWidget(self._btn_rot90)
        dr1.addWidget(self._btn_rot180); dr1.addWidget(self._btn_rot270)
        self._btn_flip_x = QPushButton('水平翻转'); self._btn_flip_y = QPushButton('垂直翻转')
        self._btn_reset = QPushButton('重置方向')
        dr2.addWidget(self._btn_flip_x); dr2.addWidget(self._btn_flip_y); dr2.addWidget(self._btn_reset)
        dl.addLayout(dr1); dl.addLayout(dr2)
        self._lbl_dir = QLabel('方向: 0°, flip=none')
        dl.addWidget(self._lbl_dir)
        ltl.addWidget(dir_group)
        self.addTab(layer_tab, '图层')

        # Tab 4: 导航
        nav_tab = QWidget()
        nl = QVBoxLayout(nav_tab)
        robot_group = QGroupBox('机器人位姿')
        rgl = QFormLayout(robot_group)
        self._lbl_robot_xy = QLabel('-'); self._lbl_robot_yaw = QLabel('-')
        rgl.addRow('坐标:', self._lbl_robot_xy); rgl.addRow('角度:', self._lbl_robot_yaw)
        nl.addWidget(robot_group)
        nav_group = QGroupBox('Nav2 状态')
        nvl = QFormLayout(nav_group)
        self._lbl_nav_status = QLabel('检查中...'); self._lbl_goal_status = QLabel('idle')
        nvl.addRow('Server:', self._lbl_nav_status); nvl.addRow('Goal:', self._lbl_goal_status)
        nl.addWidget(nav_group)
        btn_send = QPushButton('发送目标')
        btn_send.setStyleSheet('QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }')
        btn_fast = QPushButton('快速发送（谨慎）')
        btn_fast.setStyleSheet('QPushButton { background-color: #FF5722; color: white; font-weight: bold; }')
        btn_cancel = QPushButton('取消导航')
        btn_cancel.setStyleSheet('QPushButton { background-color: #f44336; color: white; }')
        btn_refresh_robot_btn = QPushButton('刷新机器位置')
        nr1, nr2 = QHBoxLayout(), QHBoxLayout()
        nr1.addWidget(btn_send); nr1.addWidget(btn_fast)
        nr2.addWidget(btn_cancel); nr2.addWidget(btn_refresh_robot_btn)
        nl.addLayout(nr1); nl.addLayout(nr2)
        nl.addStretch()
        self.addTab(nav_tab, '导航')

        # ---------- 信号连接 ----------
        btn_refresh_pts.clicked.connect(self.refresh_points.emit)
        btn_save.clicked.connect(self.save_points.emit)
        btn_add_clicked.clicked.connect(self.add_clicked.emit)
        btn_add_robot.clicked.connect(self.add_robot_pose.emit)
        btn_edit.clicked.connect(lambda: self.edit_point.emit(self._selected_index))
        btn_delete.clicked.connect(lambda: self.delete_point.emit(self._selected_index))
        btn_send.clicked.connect(lambda: self.send_goal.emit(self._selected_index, False))
        btn_fast.clicked.connect(lambda: self.send_goal.emit(self._selected_index, True))
        btn_cancel.clicked.connect(self.cancel_goal.emit)
        btn_refresh_robot_btn.clicked.connect(self.refresh_robot.emit)
        self._btn_add_route.clicked.connect(lambda: self.add_to_route.emit(self._selected_index))
        self._btn_remove_route.clicked.connect(lambda: self.remove_from_route.emit(self._route_list.currentRow()))
        self._btn_move_up.clicked.connect(lambda: self.move_route_up.emit(self._route_list.currentRow()))
        self._btn_move_down.clicked.connect(lambda: self.move_route_down.emit(self._route_list.currentRow()))
        self._btn_clear_route.clicked.connect(self.clear_route.emit)
        self._btn_start_route.clicked.connect(self.start_route.emit)
        self._btn_stop_route.clicked.connect(self.stop_route.emit)
        btn_refresh_lib.clicked.connect(self.refresh_routes_lib.emit)
        btn_preview_lib.clicked.connect(lambda: self.preview_route.emit(self._route_lib_list.currentRow()))
        btn_load_lib.clicked.connect(lambda: self.load_route_to_queue.emit(self._route_lib_list.currentRow()))
        btn_start_lib.clicked.connect(lambda: self.start_lib_route.emit(self._route_lib_list.currentRow()))
        btn_save_lib.clicked.connect(self.save_current_as_route.emit)
        btn_delete_lib.clicked.connect(lambda: self.delete_route.emit(self._route_lib_list.currentRow()))
        self._btn_team_red.clicked.connect(lambda: self.team_changed.emit('red'))
        self._btn_team_blue.clicked.connect(lambda: self.team_changed.emit('blue'))
        self._btn_record_red_start.clicked.connect(lambda: self.record_start.emit('red'))
        self._btn_record_blue_start.clicked.connect(lambda: self.record_start.emit('blue'))
        self._btn_rot0.clicked.connect(lambda: self.map_rotation.emit(0))
        self._btn_rot90.clicked.connect(lambda: self.map_rotation.emit(90))
        self._btn_rot180.clicked.connect(lambda: self.map_rotation.emit(180))
        self._btn_rot270.clicked.connect(lambda: self.map_rotation.emit(270))
        self._btn_flip_x.clicked.connect(self.map_flip_x.emit)
        self._btn_flip_y.clicked.connect(self.map_flip_y.emit)
        self._btn_reset.clicked.connect(self.map_reset_dir.emit)

    def _on_selection_changed(self, row: int):
        self._selected_index = row

    def update_point_list(self, points: list[dict]):
        """更新点位列表"""
        self._point_list.blockSignals(True)
        self._point_list.clear()
        for p in points:
            item = QListWidgetItem(f"{p['name']}  ({p['x']:.2f}, {p['y']:.2f})")
            self._point_list.addItem(item)
        self._point_list.blockSignals(False)

    def show_point_info(self, point: Optional[dict]):
        """显示选中点信息"""
        if point:
            self._lbl_name.setText(point.get('name', '-'))
            self._lbl_xy.setText(f"x={point['x']:.4f}  y={point['y']:.4f}")
            self._lbl_yaw.setText(f"{point['yaw_deg']:.2f}°")
            self._lbl_desc.setText(point.get('description', '-'))
        else:
            self._lbl_name.setText('-')
            self._lbl_xy.setText('-')
            self._lbl_yaw.setText('-')
            self._lbl_desc.setText('-')

    def update_robot_pose(self, pose: Optional[tuple]):
        """更新机器人位姿 (x, y, yaw) 或 None"""
        if pose:
            x, y, yaw = pose
            self._lbl_robot_xy.setText(f"x={x:.4f}  y={y:.4f}")
            self._lbl_robot_yaw.setText(f"{rad_to_deg(yaw):.2f}°")
        else:
            self._lbl_robot_xy.setText('不可用')
            self._lbl_robot_yaw.setText('-')

    def update_nav_status(self, available: bool, goal_state: str):
        """更新 Nav2 状态"""
        self._lbl_nav_status.setText('available' if available else 'unavailable')
        self._lbl_nav_status.setStyleSheet(
            'color: green; font-weight: bold;' if available else 'color: red; font-weight: bold;')
        self._lbl_goal_status.setText(goal_state)

    def update_costmap_status(self, global_rcv: bool, local_rcv: bool,
                              local_frame: str = ''):
        """更新 costmap 接收状态显示"""
        gs = 'received' if global_rcv else 'waiting'
        ls = 'received' if local_rcv else 'waiting'
        if local_rcv and local_frame and local_frame != 'map':
            ls = f'received ({local_frame}, limited)'
        cur = self._lbl_nav_status.text()
        # 移除旧的 costmap 状态后缀
        if ' | GCM:' in cur:
            cur = cur[:cur.index(' | GCM:')]
        self._lbl_nav_status.setText(f'{cur} | GCM: {gs} | LCM: {ls}')

    def update_route_lib_list(self, routes_data: list):
        """更新路线库列表"""
        self._route_lib_list.blockSignals(True)
        self._route_lib_list.clear()
        for r in routes_data:
            name = r.get('name', '?')
            pts = len(r.get('points', []))
            self._route_lib_list.addItem(f'{name}  ({pts} pts)')
        self._route_lib_list.blockSignals(False)

    def update_team_ui(self, team: str, start_point: Optional[dict]):
        """更新阵营 UI"""
        red_on = team == 'red'
        self._btn_team_red.setStyleSheet(
            'QPushButton { background-color: #e74c3c; color: white; font-weight: bold; }' if red_on
            else 'QPushButton { background-color: #e74c3c; color: white; }')
        self._btn_team_blue.setStyleSheet(
            'QPushButton { background-color: #3498db; color: white; font-weight: bold; }' if not red_on
            else 'QPushButton { background-color: #3498db; color: white; }')
        name = start_point['name'] if start_point else f'{team}_start'
        if start_point:
            info = (f'当前: {"红方" if red_on else "蓝方"}  |  起点: {name}\n'
                    f'x={start_point["x"]:.4f}  y={start_point["y"]:.4f}  yaw={start_point["yaw_deg"]:.2f}°')
        else:
            info = f'当前: {"红方" if red_on else "蓝方"}  |  起点: {name} (未找到)'
        self._lbl_team_info.setText(info)

    def update_dir_ui(self, rotation_str: str):
        self._lbl_dir.setText(f'方向: {rotation_str}° (view transform)')

    # ---------- 路线队列 UI ----------

    def update_route_list(self, route_points: list, current_index: int = -1,
                          route_state: str = 'idle'):
        """更新路线队列显示
        route_points: [point_dict, ...]"""
        self._route_list.blockSignals(True)
        self._route_list.clear()
        for i, pt in enumerate(route_points):
            marker = '→ ' if i == current_index else '  '
            item = QListWidgetItem(f"{marker}{i + 1}. {pt['name']}")
            if i == current_index:
                item.setForeground(QColor(0, 180, 0))
            elif i < current_index:
                item.setForeground(QColor(128, 128, 128))
            self._route_list.addItem(item)
        self._route_list.blockSignals(False)
        self._lbl_route_count.setText(f'{len(route_points)} 个点')
        self._lbl_route_state.setText(route_state)
        # 状态颜色
        colors = {'idle': '#888', 'running': '#2196F3', 'succeeded': '#4CAF50',
                  'canceled': '#FF9800', 'failed': '#f44336'}
        self._lbl_route_state.setStyleSheet(
            f'font-weight: bold; color: {colors.get(route_state, "#888")};')


# =========================================================================
# 主窗口
# =========================================================================

class MainWindow(QMainWindow):
    """GUI 主窗口"""

    def __init__(self, node: 'GoalGuiNode', map_tf: MapTransform, pgm_path: str):
        super().__init__()
        self._shutting_down = False
        self._node = node
        self._map_tf = map_tf
        self.setWindowTitle('Goal GUI — Nav2 点位导航')

        # 根据屏幕可用区域设置窗口尺寸
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry()
        default_w = min(1200, int(avail.width() * 0.90))
        default_h = min(760, int(avail.height() * 0.88))
        self.resize(default_w, default_h)
        self.setMinimumSize(800, 520)
        # 居中
        x = avail.x() + (avail.width() - default_w) // 2
        y = avail.y() + (avail.height() - default_h) // 2
        self.move(max(avail.x(), x), max(avail.y(), y))

        # 中央分割器
        splitter = QSplitter(Qt.Horizontal)

        # 左侧地图
        self._map_widget = MapWidget(map_tf, pgm_path)
        self._map_widget.setMinimumSize(400, 300)
        splitter.addWidget(self._map_widget)

        # 右侧面板（分页标签）
        self._panel = ControlPanel()
        self._panel.setMinimumWidth(280)
        self._panel.setMaximumWidth(420)
        splitter.addWidget(self._panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # 状态栏
        self._status_label = QLabel('就绪')
        self.statusBar().addPermanentWidget(self._status_label)

        # 信号连接
        self._map_widget.mouse_moved_signal.connect(self._on_map_mouse_move)
        self._map_widget.map_clicked_signal.connect(self._on_map_clicked)

        self._panel.refresh_points.connect(self._on_refresh_points)
        self._panel.save_points.connect(self._on_save_points)
        self._panel.add_clicked.connect(self._on_add_clicked)
        self._panel.add_robot_pose.connect(self._on_add_robot_pose)
        self._panel.edit_point.connect(self._on_edit_point)
        self._panel.delete_point.connect(self._on_delete_point)
        self._panel.send_goal.connect(self._on_send_goal)
        self._panel.cancel_goal.connect(self._on_cancel_goal)
        self._panel.refresh_robot.connect(self._on_refresh_robot)

        # 路线信号连接
        self._panel.add_to_route.connect(self._on_add_to_route)
        self._panel.remove_from_route.connect(self._on_remove_from_route)
        self._panel.move_route_up.connect(self._on_move_route_up)
        self._panel.move_route_down.connect(self._on_move_route_down)
        self._panel.clear_route.connect(self._on_clear_route)
        self._panel.start_route.connect(self._on_start_route)
        self._panel.stop_route.connect(self._on_stop_route)

        self._panel.layer_changed.connect(self._on_layer_changed)
        self._panel.continuous_add_toggled.connect(self._set_continuous_add)

        # 路线库信号
        self._panel.refresh_routes_lib.connect(self._on_refresh_routes_lib)
        self._panel.preview_route.connect(self._on_preview_route)
        self._panel.load_route_to_queue.connect(self._on_load_route_to_queue)
        self._panel.start_lib_route.connect(self._on_start_lib_route)
        self._panel.save_current_as_route.connect(self._on_save_current_as_route)
        self._panel.delete_route.connect(self._on_delete_route)

        self._panel.team_changed.connect(self._on_team_changed)
        self._panel.record_start.connect(self._on_record_start)

        self._panel.map_rotation.connect(self._on_map_rotation)
        self._panel.map_flip_x.connect(self._on_map_flip_x)
        self._panel.map_flip_y.connect(self._on_map_flip_y)
        self._panel.map_reset_dir.connect(self._on_map_reset_dir)

        self._panel._point_list.currentRowChanged.connect(self._on_point_selected)

        # 首次加载
        self._on_refresh_points()
        self._on_refresh_routes_lib()
        self._on_refresh_robot()
        self._on_team_changed('blue')  # 默认蓝方

        # 定时刷新
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._periodic_update)
        self._status_timer.start(500)  # 500ms

        # 应用退出前先清理 timer，避免 _periodic_update 在 ROS 半关闭时触发
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown_gui)

        self._last_clicked_map = (0.0, 0.0)
        self._continuous_add = False

        # ---------- 信号处理 ----------

    def _on_map_mouse_move(self, mx: float, my: float):
        self._status_label.setText(f'Map: x={mx:.3f}  y={my:.3f}')

    def _on_map_clicked(self, mx: float, my: float):
        self._last_clicked_map = (mx, my)
        if self._continuous_add:
            self._on_add_clicked()
        else:
            self._status_label.setText(f'已点击: x={mx:.3f}  y={my:.3f} (可用"添加点击点")')

    def _set_continuous_add(self, enabled: bool):
        self._continuous_add = enabled
        if enabled:
            self._status_label.setText('连续添加模式已开启 — 点击地图即添加点位')
        else:
            self._status_label.setText('连续添加模式已关闭')

    def _on_point_selected(self, row: int):
        points = self._node.points
        if 0 <= row < len(points):
            self._panel.show_point_info(points[row])
            # 地图高亮
            self._redraw_map(row)
        else:
            self._panel.show_point_info(None)
            self._redraw_map(None)

    def _on_refresh_points(self):
        self._node.load_points()
        self._panel.update_point_list(self._node.points)
        self._redraw_map(self._panel._selected_index)

    def _on_save_points(self):
        self._node.save_points()
        self._status_label.setText('点位已保存')

    def _on_add_clicked(self):
        mx, my = self._last_clicked_map
        suggested = self._suggest_point_name()
        dlg = PointEditDialog('添加点击点')
        dlg.x_spin.setValue(mx)
        dlg.y_spin.setValue(my)
        dlg.name_edit.setText(suggested)
        if dlg.exec_() == QDialog.Accepted:
            pt = dlg.get_point()
            if not pt['name']:
                QMessageBox.warning(self, '错误', '点位名称不能为空。')
                return
            if not self._confirm_name_overwrite(pt['name']):
                return
            self._node.add_point(pt)
            self._on_refresh_points()
            self._status_label.setText(f'已添加: {pt["name"]}')

    def _suggest_point_name(self) -> str:
        """自动生成未使用的点位名称 point_001, point_002, ..."""
        existing = {p['name'] for p in self._node.points}
        i = 1
        while True:
            name = f'point_{i:03d}'
            if name not in existing:
                return name
            i += 1

    def _confirm_name_overwrite(self, name: str) -> bool:
        """检查名称是否已存在，是则弹窗确认。返回 True 表示可继续"""
        for p in self._node.points:
            if p['name'] == name:
                reply = QMessageBox.question(
                    self, '名称冲突',
                    f'点位 "{name}" 已存在。\n\n是否覆盖旧点？\n（选 No 可返回重命名）',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                return reply == QMessageBox.Yes
        return True

    def _on_add_robot_pose(self):
        pose = self._node.get_current_pose()
        if pose is None:
            QMessageBox.warning(self, '错误', '无法获取当前机器人位姿，TF 不可用。')
            return
        x, y, yaw = pose
        suggested = self._suggest_point_name()
        dlg = PointEditDialog('添加机器人位置')
        dlg.name_edit.setText(suggested)
        dlg.x_spin.setValue(x)
        dlg.y_spin.setValue(y)
        dlg.yaw_spin.setValue(rad_to_deg(yaw))
        if dlg.exec_() == QDialog.Accepted:
            pt = dlg.get_point()
            if not pt['name']:
                QMessageBox.warning(self, '错误', '点位名称不能为空。')
                return
            if not self._confirm_name_overwrite(pt['name']):
                return
            self._node.add_point(pt)
            self._on_refresh_points()
            self._status_label.setText(f'已添加: {pt["name"]}')

    def _on_edit_point(self, index: int):
        points = self._node.points
        if not (0 <= index < len(points)):
            QMessageBox.warning(self, '提示', '请先在列表中选中一个点位。')
            return
        dlg = PointEditDialog('编辑点位', points[index].copy())
        if dlg.exec_() == QDialog.Accepted:
            pt = dlg.get_point()
            self._node.update_point(index, pt)
            self._on_refresh_points()
            self._panel._point_list.setCurrentRow(index)

    def _on_delete_point(self, index: int):
        points = self._node.points
        if not (0 <= index < len(points)):
            QMessageBox.warning(self, '提示', '请先在列表中选中一个点位。')
            return
        name = points[index]['name']
        reply = QMessageBox.question(self, '确认删除',
                                     f'确定删除点位 "{name}" 吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self._node.delete_point(index)
            self._on_refresh_points()
            self._status_label.setText(f'已删除: {name}')

    def _on_send_goal(self, index: int, skip_confirm: bool):
        points = self._node.points
        if not (0 <= index < len(points)):
            QMessageBox.warning(self, '提示', '请先在列表中选中一个点位。')
            return

        # 路线互斥：如果路线正在运行，禁止单点发送
        if self._node.route_running:
            QMessageBox.warning(self, '提示', '路线正在运行，请先停止路线再发送单点。')
            return

        pt = points[index]
        if not skip_confirm:
            reply = QMessageBox.question(
                self, '确认发送目标',
                f'是否发送目标到:\n\n'
                f'  名称: {pt["name"]}\n'
                f'  坐标: x={pt["x"]:.4f}, y={pt["y"]:.4f}\n'
                f'  角度: {pt["yaw_deg"]:.2f}°\n\n'
                f'发送？',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return

        success = self._node.send_goal(pt['name'], pt['x'], pt['y'], pt['yaw'])
        if success:
            self._status_label.setText(f'已发送目标: {pt["name"]}')
        else:
            QMessageBox.warning(self, '错误',
                                f'Action server 不可用，无法发送目标。')

    def _on_cancel_goal(self):
        if self._node.route_running:
            self._node.stop_route()
            self._status_label.setText('路线已停止')
        else:
            self._node.cancel_goal()
            self._status_label.setText('已发送取消请求')

    def _on_refresh_robot(self):
        self._update_robot_display()

    # ---------- 定时更新 ----------

    def shutdown_gui(self):
        """停止所有定时器，释放 ROS 资源（关闭窗口/退出前调用）"""
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        # 停止 _status_timer
        if hasattr(self, "_status_timer") and self._status_timer is not None:
            try:
                self._status_timer.stop()
            except Exception:
                pass
        # 查找并停止所有 QTimer 子对象
        for obj in self.findChildren(QTimer):
            try:
                obj.stop()
            except Exception:
                pass

    def _periodic_update(self):
        """500ms 定时更新"""
        if getattr(self, "_shutting_down", False):
            return
        try:
            self._update_robot_display()
            self._update_nav_status()
            self._draw_costmaps()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            if getattr(self, "_shutting_down", False):
                return
            pass

    def _update_robot_display(self):
        pose = self._node.get_current_pose()
        self._panel.update_robot_pose(pose)
        # 更新地图上的机器人箭头
        scene = self._map_widget.scene_ref
        if pose and self._panel._cb_robot.isChecked():
            x, y, yaw = pose
            px, py = self._map_tf.map_to_pixel(x, y)
            scene.set_robot_pose(px, py, yaw)
        elif scene._robot_item:
            scene.removeItem(scene._robot_item)
            scene._robot_item = None

    def _update_nav_status(self):
        available = self._node.is_action_server_ready()
        goal_state = self._node.goal_state
        self._panel.update_nav_status(available, goal_state)
        # costmap 状态
        lcm = self._node.local_costmap_msg
        self._panel.update_costmap_status(
            self._node._global_costmap_received,
            self._node._local_costmap_received,
            lcm.header.frame_id if lcm else '')

    def _redraw_map(self, selected_index: Optional[int]):
        """重绘地图上的点位"""
        scene = self._map_widget.scene_ref
        scene.remove_all_points()

        if self._panel._cb_points.isChecked():
            points = self._node.points
            start_name = f'{self._node.team_side}_start'
            for i, pt in enumerate(points):
                px, py = self._map_tf.map_to_pixel(pt['x'], pt['y'])
                scene.add_point(px, py, pt['name'], selected=(i == selected_index))
                # 起点特殊高亮
                if pt['name'] == start_name:
                    r = 6
                    from PyQt5.QtWidgets import QGraphicsEllipseItem
                    hl = QGraphicsEllipseItem(px - r, py - r, r * 2, r * 2)
                    color = QColor(255, 50, 50) if self._node.team_side == 'red' else QColor(50, 100, 255)
                    hl.setPen(QPen(color, 2))
                    hl.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 40)))
                    scene.addItem(hl)

        # 绘制路线标注
        if self._panel._cb_routes.isChecked():
            self._redraw_route_markers()
        else:
            scene._remove_all_route_markers()

    # ---------- 路线处理 ----------

    def _redraw_route_markers(self):
        """绘制路线标注到地图"""
        route = self._node.route_points
        if not route:
            self._map_widget.scene_ref._remove_all_route_markers()
            return
        pixels = []
        for pt in route:
            px, py = self._map_tf.map_to_pixel(pt['x'], pt['y'])
            pixels.append((px, py, pt['name']))
        self._map_widget.scene_ref.draw_route_markers(
            pixels, self._node.route_current_index)

    def _refresh_route_ui(self):
        self._panel.update_route_list(
            self._node.route_points,
            self._node.route_current_index,
            self._node.route_state)
        self._redraw_route_markers()

    def _on_add_to_route(self, point_index: int):
        points = self._node.points
        if not (0 <= point_index < len(points)):
            QMessageBox.warning(self, '提示', '请先在点位列表中选中一个点。')
            return
        if self._node.route_running:
            QMessageBox.warning(self, '提示', '路线正在运行，请先停止再修改。')
            return
        self._node.add_to_route(points[point_index])
        self._refresh_route_ui()

    def _on_remove_from_route(self, route_index: int):
        if self._node.route_running:
            QMessageBox.warning(self, '提示', '路线正在运行，请先停止再修改。')
            return
        self._node.remove_from_route(route_index)
        self._refresh_route_ui()

    def _on_move_route_up(self, route_index: int):
        if self._node.route_running:
            QMessageBox.warning(self, '提示', '路线正在运行，请先停止再修改。')
            return
        self._node.move_route_up(route_index)
        self._refresh_route_ui()

    def _on_move_route_down(self, route_index: int):
        if self._node.route_running:
            QMessageBox.warning(self, '提示', '路线正在运行，请先停止再修改。')
            return
        self._node.move_route_down(route_index)
        self._refresh_route_ui()

    def _on_clear_route(self):
        if self._node.route_running:
            QMessageBox.warning(self, '提示', '路线正在运行，请先停止再修改。')
            return
        self._node.clear_route()
        self._refresh_route_ui()

    def _on_start_route(self):
        if not self._node.route_points:
            QMessageBox.warning(self, '提示', '路线为空，请先加入点位。')
            return
        if self._node.route_running:
            QMessageBox.warning(self, '提示', '路线已在运行中。')
            return
        if self._node._goal_handle is not None:
            QMessageBox.warning(self, '提示', '有单点导航正在进行，请先取消再开始路线。')
            return
        pts = self._node.route_points
        names = [p['name'] for p in pts]
        reply = QMessageBox.question(
            self, '确认开始路线',
            f'是否按顺序执行 {len(pts)} 个点？\n\n'
            + ' → '.join(names) + '\n\n'
            '失败将停止整条路线。',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self._node.start_route()
        self._refresh_route_ui()
        self._status_label.setText(f'路线已开始: {len(pts)} 个点')

    def _on_stop_route(self):
        self._node.stop_route()
        self._refresh_route_ui()
        self._status_label.setText('路线已停止')

    # ---------- 图层与 costmap ----------

    def _on_layer_changed(self):
        """图层 checkbox 切换后刷新显示"""
        # 静态地图
        self._map_widget._map_item.setVisible(
            self._panel._cb_static_map.isChecked())
        # 点位 + 路线 + 机器人 在下一次 redraw 中处理
        self._redraw_map(self._panel._selected_index)

    def _draw_costmaps(self):
        """绘制 costmap overlay（500ms），所有坐标用原始 scene pixel"""
        scene = self._map_widget.scene_ref
        map_tf = self._map_tf

        def _cm_place(msg):
            if msg is None or msg.info.width <= 0 or msg.info.height <= 0:
                return
            res = msg.info.resolution
            w, h = msg.info.width, msg.info.height
            ox = msg.info.origin.position.x
            oy = msg.info.origin.position.y
            # 左上角 map 坐标
            ul_x = ox
            ul_y = oy + h * res
            # 转原始像素
            px, py = map_tf.map_to_pixel(ul_x, ul_y)
            # 缩放
            scale = res / map_tf.resolution
            raw = occupancy_to_qimage(msg, 80)
            if raw.isNull():
                return
            pix = QPixmap.fromImage(raw)
            item = QGraphicsPixmapItem(pix)
            item.setPos(QPointF(px, py))
            item.setScale(scale)
            item.setZValue(0.5)
            return item

        # global
        if self._panel._cb_global_cm.isChecked():
            gcm = self._node.global_costmap_msg
            if gcm and gcm.header.frame_id == 'map':
                item = _cm_place(gcm)
                scene.set_global_costmap_item(item)
            else:
                scene.set_global_costmap_item(None)
        else:
            scene.set_global_costmap_item(None)

        # local (odom → 暂不显示)
        scene.set_local_costmap_item(None)

    def closeEvent(self, event):
        self.shutdown_gui()
        self._node.shutdown()
        super().closeEvent(event)

    # ---------- 地图方向 ----------

    def _on_map_rotation(self, deg: int):
        self._map_widget.set_display_rotation(deg)
        self._panel.update_dir_ui(str(deg))

    def _on_map_flip_x(self):
        pass  # 暂不支持

    def _on_map_flip_y(self):
        pass  # 暂不支持

    def _on_map_reset_dir(self):
        self._map_widget.reset_view()
        self._panel.update_dir_ui('270')

    # ---------- 阵营 ----------

    def _on_team_changed(self, team: str):
        self._node.set_team(team)
        sp = self._node.get_start_point()
        self._panel.update_team_ui(team, sp)
        self._redraw_map(self._panel._selected_index)

    def _on_record_start(self, team: str):
        pose = self._node.get_current_pose()
        if pose is None:
            QMessageBox.warning(self, '错误', '无法获取当前机器人位姿，TF 不可用。')
            return
        name = f'{team}_start'
        label = '红方起点' if team == 'red' else '蓝方起点'
        rep = QMessageBox.question(self, '确认覆盖',
            f'用当前机器人位置覆盖 {label} ({name})？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if rep != QMessageBox.Yes:
            return
        x, y, yaw = pose
        pt = {
            'name': name,
            'x': round(x, 4),
            'y': round(y, 4),
            'yaw': round(yaw, 6),
            'yaw_deg': round(rad_to_deg(yaw), 2),
            'description': f'{label}（实测）',
        }
        self._node.add_point(pt)
        self._on_refresh_points()
        if self._node.team_side == team:
            sp = self._node.get_start_point()
            self._panel.update_team_ui(team, sp)
        self._status_label.setText(f'已保存: {name}')

    # ---------- 路线库处理 ----------

    def _on_refresh_routes_lib(self):
        self._node.load_routes()
        self._panel.update_route_lib_list(self._node.routes_data)

    def _on_preview_route(self, index: int):
        routes = self._node.routes_data
        if not (0 <= index < len(routes)):
            return
        route = routes[index]
        points, missing = self._node.resolve_route_points(route)
        pxs = []
        for pt in points:
            px, py = self._map_tf.map_to_pixel(pt['x'], pt['y'])
            pxs.append((px, py, pt['name']))
        self._map_widget.scene_ref.draw_route_markers(pxs, -1)
        info = f"路线: {route['name']}\n点数: {len(points)}"
        if missing:
            info += f"\n⚠ 缺失点: {', '.join(missing)}"
        self._panel._lbl_route_lib_info.setText(info)

    def _on_load_route_to_queue(self, index: int):
        routes = self._node.routes_data
        if not (0 <= index < len(routes)):
            return
        if self._node.route_running:
            QMessageBox.warning(self, '提示', '路线正在运行，请先停止再加载。')
            return
        points, missing = self._node.resolve_route_points(routes[index])
        if missing:
            QMessageBox.warning(self, '提示', f'路线有缺失点位: {", ".join(missing)}')
        self._node.route_points = list(points)
        self._refresh_route_ui()

    def _on_start_lib_route(self, index: int):
        routes = self._node.routes_data
        if not (0 <= index < len(routes)):
            return
        route = routes[index]
        points, missing = self._node.resolve_route_points(route)
        if missing:
            QMessageBox.warning(self, '错误', f'路线存在缺失点位，无法执行:\n{", ".join(missing)}')
            return
        if not points:
            QMessageBox.warning(self, '错误', '该路线无有效点位。')
            return
        if self._node.route_running:
            QMessageBox.warning(self, '提示', '路线正在运行，请先停止。')
            return
        if self._node._goal_handle is not None:
            QMessageBox.warning(self, '提示', '有单点导航正在进行，请先取消。')
            return
        reply = QMessageBox.question(
            self, '确认执行路线',
            f'是否执行预设路线 "{route["name"]}"?\n\n点数: {len(points)}\n\n'
            + ' → '.join([p['name'] for p in points]),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        self._node.route_points = list(points)
        self._node.start_route()
        self._refresh_route_ui()
        self._status_label.setText(f'路线已开始: {route["name"]}')

    def _on_save_current_as_route(self):
        if not self._node.route_points:
            QMessageBox.warning(self, '提示', '当前路线队列为空。')
            return
        name, ok = QInputDialog.getText(
            self, '保存路线模板', '路线名称:')
        if not ok or not name.strip():
            return
        name = name.strip()
        # 检查覆盖
        for i, r in enumerate(self._node.routes_data):
            if r.get('name') == name:
                rep = QMessageBox.question(self, '覆盖确认',
                    f'路线 "{name}" 已存在，覆盖？', QMessageBox.Yes | QMessageBox.No)
                if rep != QMessageBox.Yes:
                    return
                self._node.routes_data[i] = {
                    'name': name,
                    'description': '',
                    'points': [p['name'] for p in self._node.route_points],
                }
                self._node.save_routes()
                self._on_refresh_routes_lib()
                return
        self._node.routes_data.append({
            'name': name,
            'description': '',
            'points': [p['name'] for p in self._node.route_points],
        })
        self._node.save_routes()
        self._on_refresh_routes_lib()

    def _on_delete_route(self, index: int):
        routes = self._node.routes_data
        if not (0 <= index < len(routes)):
            return
        name = routes[index].get('name', '?')
        rep = QMessageBox.question(self, '确认删除',
            f'删除路线 "{name}"？', QMessageBox.Yes | QMessageBox.No)
        if rep != QMessageBox.Yes:
            return
        del self._node.routes_data[index]
        self._node.save_routes()
        self._on_refresh_routes_lib()


# =========================================================================
# ROS 节点（GUI 集成）
# =========================================================================

class GoalGuiNode(Node):
    """ROS 节点：负责 TF / Nav2 Action / 点位存储"""

    def __init__(self):
        super().__init__('goal_gui_node')

        # -------------------- 参数 --------------------
        self.declare_parameter('points_yaml', '')
        self.declare_parameter('routes_yaml', '')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('action_name', '/navigate_to_pose')

        self._points_yaml_path: str = self.get_parameter('points_yaml').value
        self._routes_yaml_path: str = self.get_parameter('routes_yaml').value
        self._map_frame: str = self.get_parameter('map_frame').value
        self._base_frame: str = self.get_parameter('base_frame').value
        self._action_name: str = self.get_parameter('action_name').value

        # 强制使用源码目录下的 YAML（不被 colcon build 覆盖）
        # 如果路径包含 /install/，替换为 /src/
        if not self._points_yaml_path or '/install/' in self._points_yaml_path:
            src_yaml = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                '..', 'config', 'goal_points.yaml')
            self._points_yaml_path = os.path.abspath(src_yaml)
        if not self._routes_yaml_path or '/install/' in self._routes_yaml_path:
            src_yaml = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                '..', 'config', 'routes.yaml')
            self._routes_yaml_path = os.path.abspath(src_yaml)

        # -------------------- 状态 --------------------
        self.points: list[dict] = []
        self.goal_state: str = 'idle'

        # Action
        self._action_client: Optional[ActionClient] = None
        self._route_action_client: Optional[ActionClient] = None
        self._goal_handle = None
        self._route_goal_handle = None
        self._active_goal_name: str = ''

        # 路线状态
        self.route_points: list[dict] = []       # 路线点列表
        self.route_running: bool = False
        self.route_current_index: int = -1
        self.route_state: str = 'idle'            # idle / running / succeeded / canceled / failed

        # 路线库
        self.routes_data: list[dict] = []          # [{'name':..., 'description':..., 'points':[...]}]

        # 阵营
        self.team_side: str = 'blue'

        # costmap 状态
        self._global_costmap_msg: Optional[OccupancyGrid] = None
        self._local_costmap_msg: Optional[OccupancyGrid] = None

        # 退出标志
        self._running = True

        # 初始化
        self._init_action_client()
        self._init_tf()
        self._init_costmap_subs()
        self.load_points()
        self.load_routes()

        self.get_logger().info(f'goal_gui_node 已启动')
        self.get_logger().info(f'点位文件: {self._points_yaml_path}')
        self.get_logger().info(f'路线文件: {self._routes_yaml_path}')
        self.get_logger().info(f'地图坐标系: {self._map_frame} → 基座: {self._base_frame}')

    def _init_action_client(self):
        self._action_client = ActionClient(self, NavigateToPose, self._action_name)
        self._route_action_client = ActionClient(
            self, NavigateThroughPoses, '/navigate_through_poses')

    def _init_tf(self):
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

    def _init_costmap_subs(self):
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
        cm_qos = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._global_cm_sub = self.create_subscription(
            OccupancyGrid, '/global_costmap/costmap',
            self._on_global_costmap, cm_qos)
        self._local_cm_sub = self.create_subscription(
            OccupancyGrid, '/local_costmap/costmap',
            self._on_local_costmap, cm_qos)
        self._global_costmap_received = False
        self._local_costmap_received = False

    def _on_global_costmap(self, msg: OccupancyGrid):
        self._global_costmap_msg = msg
        self._global_costmap_received = True

    def _on_local_costmap(self, msg: OccupancyGrid):
        self._local_costmap_msg = msg
        self._local_costmap_received = True

    @property
    def global_costmap_msg(self) -> Optional[OccupancyGrid]:
        return self._global_costmap_msg

    @property
    def local_costmap_msg(self) -> Optional[OccupancyGrid]:
        return self._local_costmap_msg

    # ---------- 点位 CRUD ----------

    def load_points(self):
        try:
            with open(self._points_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data is None:
                self.points = []
                return
            raw = data.get('points', [])
            self.points = []
            for p in raw:
                if not isinstance(p, dict):
                    continue
                self.points.append({
                    'name': str(p.get('name', '')),
                    'x': float(p.get('x', 0.0)),
                    'y': float(p.get('y', 0.0)),
                    'yaw': float(p.get('yaw', 0.0)),
                    'yaw_deg': float(p.get('yaw_deg', 0.0)),
                    'description': str(p.get('description', '')),
                })
        except FileNotFoundError:
            self.get_logger().warn(f'点位文件不存在: {self._points_yaml_path}')
            self.points = []
        except Exception as e:
            self.get_logger().error(f'加载点位失败: {e}')
            self.points = []

    def save_points(self):
        try:
            data = {'points': self.points}
            with open(self._points_yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            self.get_logger().info(f'点位已保存到 {self._points_yaml_path}')
        except Exception as e:
            self.get_logger().error(f'保存点位失败: {e}')

    # ---------- 路线库 ----------

    def load_routes(self):
        try:
            with open(self._routes_yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data is None:
                self.routes_data = []
                return
            self.routes_data = data.get('routes', [])
        except FileNotFoundError:
            self.get_logger().warn(f'路线文件不存在: {self._routes_yaml_path}')
            self.routes_data = []
        except Exception as e:
            self.get_logger().error(f'加载路线文件失败: {e}')
            self.routes_data = []

    def save_routes(self):
        try:
            data = {'routes': self.routes_data}
            with open(self._routes_yaml_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            self.get_logger().info(f'路线已保存到 {self._routes_yaml_path}')
        except Exception as e:
            self.get_logger().error(f'保存路线失败: {e}')

    def resolve_route_points(self, route: dict) -> tuple:
        """解析路线中的点名到实际点位。返回 (点位列表, 缺失点名列表)"""
        point_map = {p['name']: p for p in self.points}
        resolved = []
        missing = []
        for name in route.get('points', []):
            pt = point_map.get(name)
            if pt:
                resolved.append(pt)
            else:
                missing.append(name)
        return resolved, missing

    def get_start_point(self) -> Optional[dict]:
        """获取当前阵营起点"""
        name = f'{self.team_side}_start'
        for p in self.points:
            if p['name'] == name:
                return p
        return None

    def set_team(self, team: str):
        self.team_side = team

    def add_point(self, pt: dict):
        """添加或覆盖同名点位"""
        for i, existing in enumerate(self.points):
            if existing['name'] == pt['name']:
                self.points[i] = pt
                self.save_points()
                return
        self.points.append(pt)
        self.save_points()

    def update_point(self, index: int, pt: dict):
        if 0 <= index < len(self.points):
            self.points[index] = pt
            self.save_points()

    def delete_point(self, index: int):
        if 0 <= index < len(self.points):
            del self.points[index]
            self.save_points()

    # ---------- TF ----------

    def get_current_pose(self) -> Optional[tuple]:
        try:
            if not self._tf_buffer.can_transform(
                self._map_frame, self._base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            ):
                return None
            t = self._tf_buffer.lookup_transform(
                self._map_frame, self._base_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            x = t.transform.translation.x
            y = t.transform.translation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w
            yaw = quaternion_to_yaw(qz, qw)
            return (x, y, yaw)
        except (TransformException, Exception):
            return None

    # ---------- Nav2 Action ----------

    def is_action_server_ready(self) -> bool:
        if self._action_client is None:
            return False
        return self._action_client.server_is_ready()

    def send_goal(self, name: str, x: float, y: float, yaw: float) -> bool:
        """发送导航目标，返回是否成功提交"""
        if not self.is_action_server_ready():
            return False

        qz, qw = yaw_to_quaternion(yaw)
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self._map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self._active_goal_name = name
        self.goal_state = 'active'

        send_future = self._action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._goal_response_callback)
        return True

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f'目标被拒绝: {self._active_goal_name}')
            self.goal_state = 'failed'
            self._goal_handle = None
            return
        self._goal_handle = goal_handle
        self.goal_state = 'active'
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        result = future.result()
        status = result.status
        self._goal_handle = None
        name = self._active_goal_name
        if status == 4:
            self.goal_state = 'succeeded'
        elif status == 5:
            self.goal_state = 'canceled'
        elif status == 6:
            self.goal_state = 'failed'
        else:
            self.goal_state = f'code_{status}'
        self.get_logger().info(f'目标 "{name}" 完成: {self.goal_state}')

    def cancel_goal(self):
        if self._goal_handle is not None:
            try:
                self._goal_handle.cancel_goal_async()
                self.goal_state = 'canceled'
            except Exception as e:
                self.get_logger().error(f'取消失败: {e}')

    # ---------- 路线管理 ----------

    def add_to_route(self, pt: dict):
        self.route_points.append(dict(pt))  # shallow copy

    def remove_from_route(self, index: int):
        if 0 <= index < len(self.route_points):
            del self.route_points[index]

    def move_route_up(self, index: int):
        if 0 < index < len(self.route_points):
            self.route_points[index], self.route_points[index - 1] = \
                self.route_points[index - 1], self.route_points[index]

    def move_route_down(self, index: int):
        if 0 <= index < len(self.route_points) - 1:
            self.route_points[index], self.route_points[index + 1] = \
                self.route_points[index + 1], self.route_points[index]

    def clear_route(self):
        self.route_points.clear()
        self.route_current_index = -1
        self.route_state = 'idle'

    def start_route(self):
        if not self.route_points:
            return
        if not self._route_action_client or not self._route_action_client.server_is_ready():
            self.get_logger().error('/navigate_through_poses 不可用')
            return

        self.route_running = True
        self.route_current_index = 0
        self.route_state = 'running'

        # 构造 NavigateThroughPoses Goal
        goal_msg = NavigateThroughPoses.Goal()
        now = self.get_clock().now().to_msg()
        for pt in self.route_points:
            qz, qw = yaw_to_quaternion(pt['yaw'])
            ps = PoseStamped()
            ps.header.frame_id = self._map_frame
            ps.header.stamp = now
            ps.pose.position.x = float(pt['x'])
            ps.pose.position.y = float(pt['y'])
            ps.pose.position.z = 0.0
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            goal_msg.poses.append(ps)

        self._active_goal_name = f'route ({len(self.route_points)} pts)'
        self.get_logger().info(f'发送路线: {len(self.route_points)} 个点')
        send_future = self._route_action_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self._route_goal_response_callback)

    def _route_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('路线被拒绝')
            self.route_state = 'failed'
            self.route_running = False
            return
        self._route_goal_handle = goal_handle
        self.route_state = 'running'
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._route_goal_result_callback)

    def _route_goal_result_callback(self, future):
        result = future.result()
        status = result.status
        self._route_goal_handle = None
        self.route_running = False
        if status == 4:
            self.route_state = 'succeeded'
            self.get_logger().info('路线全部完成！')
        elif status == 5:
            self.route_state = 'canceled'
        elif status == 6:
            self.route_state = 'failed'
        else:
            self.route_state = f'code_{status}'

    def stop_route(self):
        """停止路线，取消当前 goal"""
        self.route_running = False
        self.route_state = 'canceled'
        if self._route_goal_handle is not None:
            try:
                self._route_goal_handle.cancel_goal_async()
                self._route_goal_handle = None
            except Exception:
                pass

    def shutdown(self):
        self._running = False


# =========================================================================
# 入口
# =========================================================================

def load_map_yaml(yaml_path: str) -> Optional[MapTransform]:
    """读取地图 YAML，返回 MapTransform"""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        resolution = float(data['resolution'])
        origin = data.get('origin', [0, 0, 0])
        origin_x = float(origin[0])
        origin_y = float(origin[1])

        # 读取 PGM 文件获取尺寸
        pgm_rel = data.get('image', '')
        pgm_dir = os.path.dirname(os.path.abspath(yaml_path))
        pgm_path = os.path.join(pgm_dir, pgm_rel)

        img = QImage(pgm_path)
        if img.isNull():
            # fallback: 用 PIL
            try:
                from PIL import Image
                pil_img = Image.open(pgm_path)
                w, h = pil_img.size
            except Exception:
                print(f'[ERROR] 无法读取 PGM: {pgm_path}')
                return None
        else:
            w, h = img.width(), img.height()

        return MapTransform(resolution, origin_x, origin_y, w, h), pgm_path
    except Exception as e:
        print(f'[ERROR] 加载地图 YAML 失败: {e}')
        return None


def main(args=None):
    import signal
    rclpy.init(args=args)
    node = GoalGuiNode()

    # 读取地图参数
    # 使用 os.path.realpath 解析符号链接到源码路径
    map_yaml_rel = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '..', '..', '..',
        'cod_bringup', 'maps', '2026rc.yaml')
    candidates = [
        '/home/lyu/COD26/cod_-rm2026_-navigation/src/cod_bringup/maps/2026rc.yaml',
        os.path.abspath(map_yaml_rel),
    ]
    map_tf = None
    pgm_path = None
    for c in candidates:
        if os.path.exists(c):
            result = load_map_yaml(c)
            if result:
                map_tf, pgm_path = result
                break

    if map_tf is None or pgm_path is None:
        print('[FATAL] 无法加载地图，请确认 2026rc.yaml 存在')
        node.shutdown()
        rclpy.shutdown()
        return

    # 启动 ROS spin 线程
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # 启动 Qt
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = MainWindow(node, map_tf, pgm_path)
    window.show()

    # Ctrl+C / SIGTERM 安全退出
    def _handle_signal(signum, frame):
        app.quit()
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        ret = app.exec_()
    finally:
        node.shutdown()
        executor.shutdown()
        rclpy.shutdown()
        spin_thread.join(timeout=3.0)

    sys.exit(ret)


if __name__ == '__main__':
    main()