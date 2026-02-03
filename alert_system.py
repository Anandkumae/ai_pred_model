"""
Automated Alert and Response System

Handles alerts and automated actions when AI failure is predicted.
"""

import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = "alert_config.json"
ALERT_LOG_FILE = "logs/alert_history.csv"
RETRAIN_FLAG_FILE = "flags/retrain_needed.flag"
MODEL_BACKUP_DIR = "models/backups"

class AlertSystem:
    """Manages alerts and automated responses for AI failure prediction"""
    
    def __init__(self):
        self.config = self.load_config()
        self.last_alert_time = None
        os.makedirs("logs", exist_ok=True)
        os.makedirs("flags", exist_ok=True)
        os.makedirs(MODEL_BACKUP_DIR, exist_ok=True)
    
    def load_config(self):
        """Load alert configuration"""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"{CONFIG_FILE} not found, using defaults")
            return {
                "enabled": True,
                "failure_threshold": 0.7,
                "actions": {
                    "log_alert": True,
                    "trigger_retraining": True,
                    "rollback_model": False
                }
            }
    
    def should_alert(self, failure_probability):
        """Check if alert should be triggered"""
        if not self.config.get("enabled", True):
            return False
        
        threshold = self.config.get("failure_threshold", 0.7)
        if failure_probability < threshold:
            return False
        
        # Check cooldown period
        cooldown_minutes = self.config.get("alert_cooldown_minutes", 60)
        if self.last_alert_time:
            time_since_last = datetime.now() - self.last_alert_time
            if time_since_last < timedelta(minutes=cooldown_minutes):
                logger.info(f"Alert cooldown active ({time_since_last.seconds // 60} min since last alert)")
                return False
        
        return True
    
    def send_email_alert(self, failure_probability, metrics):
        """Send email alert"""
        email_config = self.config.get("email", {})
        
        if not email_config.get("enabled", False):
            logger.info("Email alerts disabled, skipping email")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['sender']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"🚨 AI FAILURE ALERT - Probability: {failure_probability:.2%}"
            
            # Email body
            body = f"""
AI FAILURE ALERT
================

Failure Probability: {failure_probability:.2%}
Risk Level: HIGH
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Metrics:
--------
PSI Score: {metrics.get('psi_score', 'N/A')}
Avg Confidence: {metrics.get('avg_confidence', 'N/A')}
Latency: {metrics.get('latency', 'N/A')}
Concept Drift Count: {metrics.get('concept_drift_count', 'N/A')}
Error Trend: {metrics.get('error_trend', 'N/A')}

Actions Triggered:
------------------
- Alert logged
- Retraining triggered (if enabled)

Please review the system immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['sender'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent to {email_config['recipients']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def log_alert(self, failure_probability, metrics):
        """Log alert to CSV file"""
        try:
            alert_data = {
                'timestamp': datetime.now(),
                'failure_probability': failure_probability,
                'psi_score': metrics.get('psi_score', 0),
                'avg_confidence': metrics.get('avg_confidence', 0),
                'latency': metrics.get('latency', 0),
                'concept_drift_count': metrics.get('concept_drift_count', 0),
                'error_trend': metrics.get('error_trend', 0)
            }
            
            import pandas as pd
            df = pd.DataFrame([alert_data])
            
            if not os.path.exists(ALERT_LOG_FILE):
                df.to_csv(ALERT_LOG_FILE, index=False)
            else:
                df.to_csv(ALERT_LOG_FILE, mode='a', header=False, index=False)
            
            logger.info(f"Alert logged to {ALERT_LOG_FILE}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log alert: {str(e)}")
            return False
    
    def trigger_retraining(self):
        """Create flag file to trigger retraining"""
        try:
            with open(RETRAIN_FLAG_FILE, 'w') as f:
                f.write(f"Retraining triggered at {datetime.now()}\n")
            
            logger.info(f"Retraining flag created: {RETRAIN_FLAG_FILE}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create retraining flag: {str(e)}")
            return False
    
    def backup_model(self, model_path):
        """Backup current model before retraining"""
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}")
                return False
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(MODEL_BACKUP_DIR, f"model_backup_{timestamp}.pkl")
            
            import shutil
            shutil.copy2(model_path, backup_path)
            
            logger.info(f"Model backed up to {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Failed to backup model: {str(e)}")
            return False
    
    def handle_alert(self, failure_probability, metrics):
        """Main alert handler - triggers all configured actions"""
        if not self.should_alert(failure_probability):
            return {
                "alert_triggered": False,
                "reason": "Below threshold or cooldown active"
            }
        
        logger.warning(f"🚨 AI FAILURE ALERT - Probability: {failure_probability:.2%}")
        
        actions_taken = []
        
        # Log alert
        if self.config.get("actions", {}).get("log_alert", True):
            if self.log_alert(failure_probability, metrics):
                actions_taken.append("alert_logged")
        
        # Send email
        if self.config.get("email", {}).get("enabled", False):
            if self.send_email_alert(failure_probability, metrics):
                actions_taken.append("email_sent")
        
        # Trigger retraining
        if self.config.get("actions", {}).get("trigger_retraining", True):
            if self.trigger_retraining():
                actions_taken.append("retraining_triggered")
        
        # Backup and rollback model (if enabled)
        if self.config.get("actions", {}).get("rollback_model", False):
            backup_path = self.backup_model("models/traffic_primary_model.pkl")
            if backup_path:
                actions_taken.append(f"model_backed_up:{backup_path}")
        
        # Update last alert time
        self.last_alert_time = datetime.now()
        
        return {
            "alert_triggered": True,
            "failure_probability": failure_probability,
            "actions_taken": actions_taken,
            "timestamp": self.last_alert_time.isoformat()
        }

# Global instance
alert_system = AlertSystem()

if __name__ == "__main__":
    # Test alert system
    print("Testing Alert System...")
    
    test_metrics = {
        'psi_score': 0.35,
        'avg_confidence': 0.52,
        'latency': 0.08,
        'concept_drift_count': 4,
        'error_trend': 0.025
    }
    
    result = alert_system.handle_alert(0.85, test_metrics)
    print(json.dumps(result, indent=2))
