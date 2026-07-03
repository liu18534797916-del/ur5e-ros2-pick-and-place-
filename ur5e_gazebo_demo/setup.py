from setuptools import setup
from glob import glob
import os

package_name = 'ur5e_gazebo_demo'

setup(
    name=package_name,
    version='0.0.0',

    packages=[package_name],

    data_files=[

        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),

        (
            'share/' + package_name,
            ['package.xml']
        ),

        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')
        ),

        (
            os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')
        ),

        (
            os.path.join('share', package_name, 'worlds'),
            glob('worlds/*')
        ),

        (
            os.path.join('share', package_name, 'config'),
            glob('config/*')
        ),

        (
            os.path.join('share', package_name, 'models', 'aruco_marker'),
            glob('models/aruco_marker/*')
        ),
        (
            os.path.join('share', package_name, 'models', 'box_1'),
            glob('models/box_1/*')
        ),
        (
            os.path.join('share', package_name, 'models', 'box_2'),
            glob('models/box_2/*')
        ),
        (
            os.path.join('share', package_name, 'models', 'box_3'),
            glob('models/box_3/*')
        ),

        (
            os.path.join('share', package_name, 'materials', 'textures'),
            glob('materials/textures/*')
        ),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='shuai',
    maintainer_email='shuai@example.com',

    description='UR5e Gazebo Demo',
    license='Apache License 2.0',

    tests_require=['pytest'],

    entry_points={
        'console_scripts': [

            'pick_and_place_moveit = ur5e_gazebo_demo.pick_and_place_moveit:main',

            'vision_pick = ur5e_gazebo_demo.vision_pick:main',

            'aruco_detect = ur5e_gazebo_demo.aruco_detect:main',

            'calibrate_camera = ur5e_gazebo_demo.calibrate_camera:main',

            'camera_tf_publisher = ur5e_gazebo_demo.camera_tf_publisher:main',

            'object_pose_estimate = ur5e_gazebo_demo.object_pose_estimate:main',

            'spawn_boxes = ur5e_gazebo_demo.spawn_boxes:main',

        ],
    },
)