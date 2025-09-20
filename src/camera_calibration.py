import numpy as np
import cv2

from typing import List, Tuple, Dict, Optional, Union

class CameraCalibrator:
    def __init__(self, image_size: Tuple[int, int]):
        """
        Initialize the camera calibrator with image dimensions.
        
        Args:
            image_size: (width, height) of the images used for calibration
        """
        self.image_size = image_size
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reprojection_error = None
    
    def calibrate_from_points(
        self,
        object_points_list: List[np.ndarray],  # List of Nx3 points in world coordinates
        image_points_list: List[np.ndarray],   # List of Nx2 points in image coordinates
        flags: int = 0
    ) -> bool:
        """
        Calibrate camera using 2D-3D point correspondences.
        
        Args:
            object_points_list: List of Nx3 numpy arrays containing 3D points in world coordinates
            image_points_list: List of Nx2 numpy arrays containing corresponding 2D points in image coordinates
            flags: Calibration flags (see OpenCV docs for calibrateCamera)
            
        Returns:
            bool: True if calibration was successful, False otherwise
        """
        if len(object_points_list) != len(image_points_list):
            raise ValueError("object_points_list and image_points_list must have the same length")
        
        if len(object_points_list) < 3:
            raise ValueError("At least 3 sets of point correspondences are required")
        
        # Convert all points to float32 if they aren't already
        object_points_list = [p.astype(np.float32) for p in object_points_list]
        image_points_list = [p.astype(np.float32) for p in image_points_list]
        
        # Initialize camera matrix
        focal_length = self.image_size[1]
        cx, cy = self.image_size[0]/2, self.image_size[1]/2
        camera_matrix = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Initialize distortion coefficients
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points_list,
            image_points_list,
            self.image_size,
            camera_matrix,
            dist_coeffs,
            flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        
        if not ret:
            return False
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(object_points_list)):
            imgpoints2, _ = cv2.projectPoints(
                object_points_list[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(image_points_list[i], imgpoints2.reshape(-1, 2), cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        # Store results
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs.ravel()
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.reprojection_error = mean_error / len(object_points_list)
        
        return True
    
    def estimate_pose(
        self,
        object_points: np.ndarray,  # Nx3 points in world coordinates
        image_points: np.ndarray,   # Nx2 points in image coordinates
        use_extrinsic_guess: bool = False,
        flags: int = cv2.SOLVEPNP_ITERATIVE
    ) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Estimate the pose of an object given 3D-2D point correspondences.
        
        Args:
            object_points: Nx3 numpy array of 3D points in world coordinates
            image_points: Nx2 numpy array of corresponding 2D points in image coordinates
            use_extrinsic_guess: If True, use current rvec and tvec as initial guess
            flags: PnP method flag (see OpenCV solvePnP flags)
            
        Returns:
            Tuple of (success, rvec, tvec)
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError("Camera must be calibrated first")
        
        if len(object_points) < 4:
            raise ValueError("At least 4 points are required for PnP")
        
        # Convert points to float32 if they aren't already
        object_points = object_points.astype(np.float32)
        image_points = image_points.astype(np.float32)
        
        # Initialize rvec and tvec if needed
        rvec = np.zeros(3, dtype=np.float64) if not use_extrinsic_guess else self.rvecs[-1]
        tvec = np.zeros(3, dtype=np.float64) if not use_extrinsic_guess else self.tvecs[-1]
        
        # Solve PnP
        ret, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=use_extrinsic_guess,
            flags=flags
        )
        
        if not ret:
            return False, None, None
            
        return True, rvec, tvec
    
    def project_points(
        self,
        object_points: np.ndarray,  # Nx3 points in world coordinates
        rvec: np.ndarray = None,
        tvec: np.ndarray = None
    ) -> np.ndarray:
        """
        Project 3D points to 2D image plane.
        
        Args:
            object_points: Nx3 numpy array of 3D points in world coordinates
            rvec: Rotation vector (if None, uses identity rotation)
            tvec: Translation vector (if None, uses zero translation)
            
        Returns:
            Nx2 numpy array of projected 2D points
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise RuntimeError("Camera must be calibrated first")
            
        if rvec is None:
            rvec = np.zeros(3, dtype=np.float64)
        if tvec is None:
            tvec = np.zeros(3, dtype=np.float64)
            
        object_points = object_points.astype(np.float32)
        
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs
        )
        
        return projected.reshape(-1, 2)
    
    def get_calibration_results(self) -> Dict:
        """
        Get the calibration results.
        
        Returns:
            Dictionary containing calibration parameters
        """
        if self.camera_matrix is None:
            raise RuntimeError("Camera is not calibrated")
            
        return {
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'image_size': self.image_size,
            'reprojection_error': self.reprojection_error,
            'num_views': len(self.rvecs) if self.rvecs is not None else 0
        }
    
    def save_calibration(self, filename: str) -> bool:
        """
        Save calibration parameters to a file.
        
        Args:
            filename: Path to save the calibration file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.camera_matrix is None:
            return False
            
        try:
            np.savez_compressed(
                filename,
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                image_size=np.array(self.image_size, dtype=np.int32),
                reprojection_error=self.reprojection_error,
                num_views=len(self.rvecs) if self.rvecs is not None else 0
            )
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    @classmethod
    def load_calibration(cls, filename: str) -> 'CameraCalibrator':
        """
        Load calibration from a file and return a new CameraCalibrator instance.
        
        Args:
            filename: Path to the calibration file
            
        Returns:
            A new CameraCalibrator instance with loaded calibration
        """
        try:
            data = np.load(filename, allow_pickle=False)
            calib = cls(tuple(data['image_size']))
            calib.camera_matrix = data['camera_matrix']
            calib.dist_coeffs = data['dist_coeffs']
            calib.reprojection_error = float(data['reprojection_error'])
            return calib
        except Exception as e:
            print(f"Error loading calibration: {e}")
            raise

# Example usage
if __name__ == "__main__":
    # Example: Create a synthetic calibration scenario
    # In practice, you would load your actual 3D-2D point correspondences
    
    # Image size (width, height)
    image_size = (1920, 1080)
    
    # Create a calibrator instance
    calibrator = CameraCalibrator(image_size)
    
    # Example: Generate synthetic 3D points (in mm)
    # These would be your known 3D points in world coordinates
    num_points = 20
    object_points = np.random.rand(num_points, 3) * 100  # Random points in a 100x100x100 mm cube
    
    # Example: Generate corresponding 2D points (in pixels)
    # In practice, these would be your detected points in the image
    # For this example, we'll project the 3D points using a known camera matrix
    camera_matrix = np.array([
        [1500, 0, image_size[0]/2],
        [0, 1500, image_size[1]/2],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.array([-0.2, 0.1, 0, 0, 0], dtype=np.float64)
    
    # Generate multiple views by rotating and translating the object
    num_views = 5
    object_points_list = []
    image_points_list = []
    
    for i in range(num_views):
        # Create a random rotation and translation
        rvec = np.random.rand(3) * np.pi / 4  # Random rotation up to 45 degrees
        tvec = np.array([0, 0, 500 + i*100])  # Move along Z axis
        
        # Project 3D points to 2D
        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist_coeffs
        )
        
        # Add some noise to simulate detection error
        image_points = image_points.reshape(-1, 2) + np.random.normal(0, 0.5, (num_points, 2))
        
        object_points_list.append(object_points)
        image_points_list.append(image_points)
    
    # Now calibrate using the generated points
    success = calibrator.calibrate_from_points(object_points_list, image_points_list)
    
    if success:
        print("Calibration successful!")
        calib_results = calibrator.get_calibration_results()
        print(f"Camera matrix:\n{calib_results['camera_matrix']}")
        print(f"Distortion coefficients: {calib_results['dist_coeffs']}")
        print(f"Reprojection error: {calib_results['reprojection_error']:.5f} pixels")
        
        # Save calibration
        calibrator.save_calibration("camera_calibration_points.npz")
        
        # Example of loading the calibration
        loaded_calib = CameraCalibrator.load_calibration("camera_calibration_points.npz")
        print("\nLoaded calibration:")
        print(f"Image size: {loaded_calib.image_size}")
        print(f"Reprojection error: {loaded_calib.reprojection_error:.5f} pixels")
