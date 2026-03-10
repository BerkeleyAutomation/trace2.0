from yumi_jacobi.interface import Interface
from jacobi import JacobiError
import time
import asyncio

async def main():
    interface = Interface(speed=1.0, file='/home/justinyu/multicable-decluttering/yumi_jacobi/starter_examples/AUTOLAB_BWW_YuMi.jacobi-project')
    interface.home()
    interface.calibrate_grippers()
    leftJA = interface.get_joint_positions('left')
    rightJA = interface.get_joint_positions('right')
    print(leftJA)
    print(rightJA)
    
    counter = 0
    while counter < 40:
        try:
            # increment leftJA and rightJA by one degree
            if counter < 20:
                leftJA = [ja + 0.001 for ja in leftJA]
                rightJA = [ja + 0.001 for ja in rightJA]
            else:
                leftJA = [ja - 0.001 for ja in leftJA]
                rightJA = [ja - 0.001 for ja in rightJA]
                
            start_time = time.time()
            traj = interface._async_interface.planner.plan(start={
            interface._async_interface.yumi.left: interface._async_interface.driver_left.current_joint_position,
            interface._async_interface.yumi.right: interface._async_interface.driver_right.current_joint_position,
            },
            goal={
                interface._async_interface.yumi.left: leftJA,
                interface._async_interface.yumi.right: rightJA,
            },
            )
            # print(f"Planning time: {(time.time() - start_time)*1000} ms")
            start_time = time.time()
            await interface._async_interface.driver_left.run_async(traj)
            print(f"Time to send trajectory: {(time.time() - start_time)*1000} ms")
            # await interface._async_interface.driver_right.run_async(traj)
            print(f"Total deploy-to-finish time: {(time.time() - start_time)*1000} ms")
            print(f"Frequency: {1/(time.time() - start_time)} Hz")
            counter += 1
        except JacobiError as e:
            print(e)
            break
        
    counter = 0
    while counter < 40:
        try:
            # increment leftJA and rightJA by one degree
            if counter < 20:
                leftJA = [ja + 0.001 for ja in leftJA]
                rightJA = [ja + 0.001 for ja in rightJA]
            else:
                leftJA = [ja - 0.001 for ja in leftJA]
                rightJA = [ja - 0.001 for ja in rightJA]
            start_time = time.time()
            interface._async_interface.driver_left.move_to_async(leftJA, ignore_collisions=True)
            await interface._async_interface.driver_right.move_to_async(rightJA, ignore_collisions=True)
            print(f"Time: {(time.time() - start_time)*1000} ms")
            print(f"Frequency: {1/(time.time() - start_time)} Hz")
            counter += 1
        except JacobiError as e:
            print(e)
            break
                
        

if __name__ == '__main__':
    asyncio.run(main())