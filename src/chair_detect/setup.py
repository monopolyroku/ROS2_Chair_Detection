from setuptools import setup

package_name = 'chair_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='niloclol',
    maintainer_email='2101737@sit.singaporetech.edu.sg',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 
            'chair_detect = chair_detect.chair_detect_v3:main',
            'listener = chair_detect.listener:main'
        ],
    },
)


