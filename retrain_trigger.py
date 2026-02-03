"""
Automated Retraining Trigger

Monitors for retraining flag and automatically retrains the meta-model
when AI failure is predicted.
"""

import os
import time
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
RETRAIN_FLAG_FILE = "flags/retrain_needed.flag"
CHECK_INTERVAL_SECONDS = 60  # Check every minute

def retrain_model():
    """Execute retraining pipeline"""
    logger.info("=" * 70)
    logger.info("AUTOMATED RETRAINING TRIGGERED")
    logger.info("=" * 70)
    
    try:
        # Step 1: Generate fresh failure dataset
        logger.info("Step 1: Generating failure dataset...")
        os.system("python generate_failure_dataset.py")
        
        # Step 2: Label the dataset
        logger.info("Step 2: Labeling dataset...")
        os.system("python label_failures.py")
        
        # Step 3: Train new meta-model
        logger.info("Step 3: Training meta-model...")
        os.system("python train_failure_model.py")
        
        logger.info("✓ Retraining completed successfully")
        logger.info("=" * 70)
        
        return True
        
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        return False

def monitor_retrain_flag():
    """Monitor for retraining flag and trigger retraining"""
    logger.info("Retraining monitor started")
    logger.info(f"Checking for flag: {RETRAIN_FLAG_FILE}")
    logger.info(f"Check interval: {CHECK_INTERVAL_SECONDS} seconds")
    
    while True:
        try:
            if os.path.exists(RETRAIN_FLAG_FILE):
                logger.warning(f"🚨 Retraining flag detected!")
                
                # Read flag file
                with open(RETRAIN_FLAG_FILE, 'r') as f:
                    flag_content = f.read()
                logger.info(f"Flag content: {flag_content.strip()}")
                
                # Remove flag file
                os.remove(RETRAIN_FLAG_FILE)
                logger.info("Flag file removed")
                
                # Trigger retraining
                success = retrain_model()
                
                if success:
                    logger.info("✓ Automated retraining successful")
                else:
                    logger.error("✗ Automated retraining failed")
            
            # Wait before next check
            time.sleep(CHECK_INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            logger.info("Retraining monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Monitor error: {str(e)}")
            time.sleep(CHECK_INTERVAL_SECONDS)

if __name__ == "__main__":
    print("=" * 70)
    print("AUTOMATED RETRAINING MONITOR")
    print("=" * 70)
    print(f"Monitoring: {RETRAIN_FLAG_FILE}")
    print(f"Interval: {CHECK_INTERVAL_SECONDS} seconds")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    monitor_retrain_flag()
