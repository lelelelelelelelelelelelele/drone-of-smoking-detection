import cv2
import numpy as np

class Aruco_detector:
    '''
    A class to represent a camera calibration setup with ARUCO marker detection capabilities.
    
    Attributes:
        aruco_dict_type (str): Type of the ARUCO dictionary to be used for marker detection.
        intrinsic_camera (numpy.ndarray): Intrinsic parameters of the camera.
        distortion (numpy.ndarray): Distortion coefficients of the camera.
    '''
    
    def __init__(self, aruco_dict_type, intrinsic_camera, distortion):
        '''
        Initializes the CameraCalibrationSetup instance with given parameters.
        
        Args:
            aruco_dict_type (str): String specifying the ARUCO dictionary type (e.g., 'DICT_4X4_50').
            intrinsic_camera (numpy.ndarray): Matrix containing the intrinsic camera parameters.
            distortion (numpy.ndarray): Vector containing the distortion coefficients.
            parameter
        '''
        self.dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        # Validate and set intrinsic camera parameters
        if not isinstance(intrinsic_camera, np.ndarray) or intrinsic_camera.shape != (3, 3):
            raise ValueError('intrinsic_camera must be a 3x3 numpy.ndarray.')
        self.intrinsic_camera = intrinsic_camera
        # Validate and set distortion coefficients
        if not isinstance(distortion, np.ndarray) or distortion.size != 5:
            raise ValueError('distortion must be a 1D numpy.ndarray of length 5.')
        self.distortion = distortion
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
    def undistort_image(self, distorted_image):
        '''
        Undistorts an image using the camera's intrinsic parameters and distortion coefficients.
        
        Args:
            distorted_image (numpy.ndarray): The distorted image to be corrected.
        
        Returns:
            numpy.ndarray: The undistorted image.
        '''
        return cv2.undistort(distorted_image, self.intrinsic_camera, self.distortion)

    def detect_markers(self, image, number):
        '''
        Detects ARUCO markers in an image using the configured ARUCO dictionary.
        
        Args:
            image (numpy.ndarray): The input image where markers are to be detected.
        
        Returns:
            corners, ids, rejectedImgPoints: Tuple containing corners of the markers, their IDs, and rejected points.
        '''
        image = self.undistort_image(image)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)# 进行ArUco检测返回对应结果
        detected = -1
        center_x, center_y = 0, 0
        if len(corners) > 0:
            ids = ids.flatten()
            
            for (markerCorner, markerID) in zip(corners, ids):
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(image, bottomLeft, topLeft, (0, 255, 0),2)        
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)       
                cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                	0.5, (0, 255, 0), 2)
                print("[Inference] ArUco marker ID: {}".format(markerID))
                if number == markerID or number == -1:
                    detected = markerID
                    center_x, center_y = cX, cY

        return image, detected, center_x, center_y

