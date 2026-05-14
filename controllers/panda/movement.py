import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3

class PandaMovement:
    POSITION_THRESHOLD = 0.002
    CONFIDENCE_THRESHOLD = 0.2

    def __init__(self, robot,time_step):
        self.robot = robot
        self.panda = rtb.models.DH.Panda()                      # Panda model for inverse kinematics
        self.time_step = time_step                              # Time step for moving forward the simulation
        self.motors = []                                        # Motor's nodes
        self.sensors = []                                       # Positional sensors of the different motors
        self.target_joint_pos_is_calc = False                   # Boolean value to indicate that no computations have to be performed
        self.calc_target_joint_pos = []                         # Target position for each joint
        return
    
    # Initialization of the Panda joints and sensors
    def init_arm_components(self):

        self.motors = []
        self.sensors = []

        # Recover joints and sensors for moving the arm
        for i in range(7):
            motor = self.robot.getDevice("panda_joint" + str(i+1))
            motor.setVelocity(1)
            self.motors.append(motor)

            sensor = motor.getPositionSensor()
            sensor.enable(self.time_step)
            self.sensors.append(sensor)

                # Recover hand joints
        lfinger = self.robot.getDevice("panda_finger::left")
        lfinger.setVelocity(0.2)
        self.motors.append(lfinger)
        sensor = lfinger.getPositionSensor()
        sensor.enable(self.time_step)
        self.sensors.append(sensor)

        rfinger = self.robot.getDevice("panda_finger::right")
        rfinger.setVelocity(0.2)
        self.motors.append(rfinger)
        sensor = rfinger.getPositionSensor()
        sensor.enable(self.time_step)
        self.sensors.append(sensor)
        return
    
    # Logging function for debugging
    def log(self, message):
        """ Print a log message in the console if the debug mode is enabled. """
        print(f"[PANDA] {message}")

    # Move the arm to the given position
    def rotate_back(self, time_limit):

        # Set the positions
        self.motors[0].setPosition(2.88)

        # Perform the movement until it is finished
        while (self.is_position_reached(self.motors[0], 2.88) == False):
            self.robot.step(self.time_step)

    # Move the arm to the given position
    def move_arm(self, final_position, time_limit):
        curr_cycles = 0
        num_cycles = 1
        ret = 0

        self.log(f"Moving the arm to position: {final_position}")
        
        # Compute the new joints positions only once
        if self.target_joint_pos_is_calc == False:
            self.compute_movement(final_position)

        # Compute how many cycles before stopping due to the imposed time limit
        if time_limit != 0:
            curr_cycles = 1
            num_cycles = int(time_limit / self.time_step)

        # Set the positions
        for i in range(7):
            self.motors[i].setPosition(self.calc_target_joint_pos[i])

        # Perform the movement until it is finished
        for i in range(7):
            while (self.is_position_reached(self.motors[i], self.calc_target_joint_pos[i]) == False and curr_cycles < num_cycles):
                self.robot.step(self.time_step)
                if time_limit != 0:
                    curr_cycles = curr_cycles + 1

        # Need to distinguish return value since the movement can be of partial type
        ret = 0
        for i in range(7):
            if self.is_position_reached(self.motors[i], self.calc_target_joint_pos[i]) == False:
                ret = 1
                
        # Reset the flag
        self.target_joint_pos_is_calc = False

        return ret


    # Open/close the gripper
    def control_gripper(self, action):
        """
        Open or close the gripper.
        
        Parameters:
        action (str): The action to be performed. Use 'open' or 'close'.
        """
        if action not in ["open", "close"]:
            raise ValueError("Invalid action. Use 'open' or 'close'.")
        
        # Distiguish the target position
        if action == "open":
            if self.debug:
                self.log("Opening the gripper...")
            target = 0.04

        elif action == "close":
            if self.debug:
                self.log("Closing the gripper...")
            target = 0.017
        
        # Set the positions
        for i in range(7, 9):
            self.motors[i].setPosition(target)
        
        # Perform the movement
        for i in range(7, 9):
            while (self.is_position_reached(self.motors[i], target) == False):
                self.robot.step(self.time_step)
                
        return

    # Compute the trajectory to reach target endpoint
    def compute_movement(self, final_position):
        """ Before moving the arm to the pointed location, the exact angle of each joint
            has to be computed. To do this, the model of the Panda robotic arm from the roboticstoolbox
            library is harnessed. For the values to be correct, the current positions of the joints have to be provided though. """
        arm_pos = []
        self.calc_target_joint_pos = []

        for i in range(7):
            sensor = self.sensors[i]
            arm_pos.append(sensor.getValue())

        self.panda.q = np.array(arm_pos)

        current_position = self.robot.getSelf().getField("translation").getSFVec3f()
        x = final_position[0] - current_position[0]
        y = final_position[1] - current_position[1]
        z = final_position[2] - current_position[2]

        # Compute transformation matrix
        T_matrix = SE3.Trans(x, y, z) * SE3.OA([0, 1, 0], [0, 0, -1])
        
        # Search for a valid solution to the inverse kinematic problem
        not_valid = True
        while not_valid == True:
            not_valid = False
            inverse_kin_sol = (self.panda.ikine_LM(T_matrix)).q
            for i in range(7):
                if not (self.motors[i].getMinPosition() < inverse_kin_sol[i] and self.motors[i].getMaxPosition() > inverse_kin_sol[i]):
                    not_valid = True

        for i in range(7):
            self.target_joint_pos_is_calc = True
            self.calc_target_joint_pos.append(inverse_kin_sol[i])

    # Check if a joint has reached the requested position
    def is_position_reached(self, motor, pos):
        return True if ((abs(motor.getPositionSensor().getValue() - pos)) <= self.POSITION_THRESHOLD) else False
    