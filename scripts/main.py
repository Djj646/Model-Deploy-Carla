import torch
import torch.nn as nn
from torch.autograd import grad
from PIL import Image
import torchvision.transforms as transforms
from learning.model_train_path import TrajFrom2Pic

# sys.path为环境列表模块，可动态修改解释器和相关环境
from os.path import join, dirname
import sys

# 0代表第一优先级，insert添加指定路径到导入模块的搜索文件夹，__file__返回相对路径（在sys.path中时），".."表示上第二级目录
sys.path.insert(0, join(dirname(__file__), '..'))

# 自定义的simulator和util配置模块
import simulator

# ./CarlaUE4.sh 路径
simulator.load('D:/CARLA_0.9.13/WindowsNoEditor')
# PythonAPI中的carla模块
import carla
from agents.navigation.controller import VehiclePIDController

sys.path.append('D:/CARLA_0.9.13/WindowsNoEditor/PythonAPI/carla')
from agents.navigation.basic_agent import BasicAgent
from simulator import config, set_weather, add_vehicle
from simulator.sensor_manager import SensorManager
from utils.navigator_sim import get_random_destination, get_map, get_nav, replan, close2dest

import os
import cv2
import time
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm

from ff.capac_controller import CapacController

# 加载模型
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = TrajFrom2Pic().to(device)
model_dir="models/img2path-01/model_2000.pth"
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint)

# 参数剖析器
# 设定data index目录索引默认为1，总数total number默认为10000
parser = argparse.ArgumentParser(description='Params')
parser.add_argument('-n', '--num', type=int, default=100000, help='total number')
parser.add_argument('--max_t', type=float, default=3., help='max time')
args = parser.parse_args()

global_img = None
global_nav = None
global_vel = None
MAX_SPEED = 30

# 当前决策所需信息
# img, nav, t, v_0
batch_now={
    'img': global_img,
    'routine': global_nav,
    't': 0,
    'v_0': 0
}

# 图片简单处理（未使用）
# 原cost_map
transforms_ = [ transforms.Resize((200, 400), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            ]
img_trans = transforms.Compose(transforms_)

# 相机反馈函数 将相机data存入全局变量global_img中，并转化成tensor
def image_callback(data):
    global global_img
    array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    # (samples, channels, height, width)
    # 4 channels: B, G, R, A
    array = np.reshape(array, (data.height, data.width, 4))  # BGRA format
    array = np.stack([array[2, :], array[1, :], array[0, :]], axis=0) # RGB format
    
    # 需要用图片处理成200*400格式（插值）
    img = Image.fromarray(array)
    cv2.imshow('Vision', img)
    
    global_img = img_trans(img)
    # global_img = torch.from_numpy(array)
    

def get_vel(actor):
    global_vel = actor.get_velocity()
    _vx = global_vel.x
    _vy = global_vel.y
    v_0 = np.sqrt(_vx*_vx + _vy*_vy)
    
    return v_0
    # velocity处理

def main():
    # 全局变量 卫星导航图  速度
    global global_nav,  global_vel

    # 从字典config参数表设置client（到server）
    # 构造函数Client(self,host(IP),port(TCP port),worker_threads)
    client = carla.Client(config['host'], config['port'])
    client.set_timeout(config['timeout'])

    # world = client.get_world()
    # client请求创建一个新世界，load_word(map_name)
    world = client.load_world('Town01')
    """
    weather = carla.WeatherParameters(
        cloudiness=random.randint(0,50),
        precipitation=0,
        sun_altitude_angle=random.randint(40,90)
    )
    set_weather(world, weather)
    """

    # 设置天气为晴天
    # WeatherParameters见P82
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # 返回生成的角色的蓝图
    blueprint = world.get_blueprint_library()

    # 包含道路信息和航点管理的类carla.map
    world_map = world.get_map()

    # 在simulator构造函数中封装好的生成车辆函数，返回carla.Actor
    vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.audi.a2')
    # Enables or disables the simulation of physics on this actor.
    # 启用物理模拟
    vehicle.set_simulate_physics(True)

    # 传感器配置参数字典
    # 配置项：传感器安装位置 数据保存的响应函数名
    sensor_dict = {
        'camera': {
            'transform': carla.Transform(carla.Location(x=0.5, y=0.0, z=2.5)),
            'callback': image_callback,
        },
    }
    # SensorManager::__init__(self, world, blueprint, vehicle, param_dict)
    # 传感器安置与初始化
    sm = SensorManager(world, blueprint, vehicle, sensor_dict)
    sm.init_all()
    time.sleep(0.3)

    # 返回地图创建者的建议点位表用于车辆生成
    spawn_points = world_map.get_spawn_points()

    # 返回openDRIVE文件的拓扑的最小图元祖列表
    waypoint_tuple_list = world_map.get_topology()
    origin_map = get_map(waypoint_tuple_list)

    # 使用carla自带的行为规划器进行导航
    # 但不实施runstep()控制
    agent = BasicAgent(vehicle, target_speed=MAX_SPEED)

    # port = 8000
    # tm = client.get_trafficmanager(port)
    # vehicle.set_autopilot(True,port)
    # tm.ignore_lights_percentage(vehicle,100)

    # 随机目的地
    destination = get_random_destination(spawn_points)
    # 全局导航图（用于裁剪nav）
    plan_map = replan(agent, destination, copy.deepcopy(origin_map))
    
    # 控制对象，通过CapacController确定
    ctrller = CapacController(world, vehicle, 30)
    control = carla.VehicleControl()

    # 迭代
    for cnt in tqdm(range(args.num)):
        if close2dest(vehicle, destination):
            print('get destination !', time.time())
            destination = get_random_destination(spawn_points)
            plan_map = replan(agent, destination, copy.deepcopy(origin_map))

        if vehicle.is_at_traffic_light():
            traffic_light = vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)

        # 实时获取位置
        x = vehicle.get_location().x
        y = vehicle.get_location().y
        
        # 实时获取RGB
        # 见相机反馈函数
        # img需经过transform处理
        global_img = global_img.to(device)
        global_img.requires_grad = True

        # 实时获取nav
        # nav需经过transform处理
        nav = get_nav(vehicle, plan_map)
        # cv2.imshow('Nav', global_nav)
        global_nav = img_trans(nav)
        global_nav.requires_grad = True
        
        # 实时获取t
        t = torch.arange(0, 0.99, args.dt).unsqueeze(1).to(device)
        t.requires_grad = True
        batch_now['t'] = t

        # 实时获取v_0
        v_0 = get_vel(vehicle)
        batch_now['v_0'] = v_0

        # 等待数据，开启子线程plan_thread
        # while True:
        #     if (global_img is not None) and (global_nav is not None):
        #         # 规划线程
        #         plan_thread.start()
        #         break
        #     else:
        #         time.sleep(0.001)
        
        # 模型预测轨迹
        output = model(batch_now['img'], batch_now['routine'], batch_now['t'], batch_now['v_0'])
        
        # 控制信息处理
        # vx,vy -> ax,ay -> control(throttle, steer)
        # 求梯度需要准备集合？
        vx = grad(output[:, 0].sum(), batch_now['t'], create_graph=True)[0]
        vy = grad(output[:, 1].sum(), batch_now['t'], create_graph=True)[0]
        
        ax = grad(vx.sum(), batch_now['t'], create_graph=True)[0]
        ay = grad(vy.sum(), batch_now['t'], create_graph=True)[0]
        
        output_axy = torch.cat([ax, ay], dim=1)
        
        theta_a = torch.atan2(ay, ax)
        theta_v = torch.atan2(vy, vx)
        sign = torch.sign(torch.cos(theta_a-theta_v))
        a = torch.mul(torch.norm(output_axy, dim=1), sign.flatten()).unsqueeze(1)

        trajectory = {'time':time.time(), 'x':x, 'y':y, 'vx':vx, 'vy':vy, 'ax':ax, 'ay':ay, 'a':a}
        
        # control主要可控参数
        # throttle(油门): [0.0, 1.0]; steer(方向盘): [-1.0, 1.0]; brake(脚刹): [0.0, 1.0]
        control.manual_gear_shift = False

        # 基本控制思路：根据vx, vy进行PID控制
        
        #--- 以下参考examples/manual_control.py ---
        # 实施控制
        # 当前控制时间
        control_time = time.time()
        dt = control_time - trajectory['time']
        index = int((dt/args.max_t)//args.dt)
        
        if index > 0.99/args.dt:
            continue
        
        control = ctrller.run_step(trajectory, index, state0)
        vehicle.apply_control(control)
        
        # 查看执行情况
        cv2.waitKey(10)

    cv2.destroyAllWindows()
    sm.close_all()
    vehicle.destroy()


# 若作为包使用则不运行main()，__name__反映模块层次
if __name__ == '__main__':
    main()