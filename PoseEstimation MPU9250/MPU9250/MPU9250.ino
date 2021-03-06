#include "MPU9250.h"
#include "SensorFusion.h"
#include "math.h"

SF fusion;

//Sensor measures values
float ax, ay, az;
float gx, gy, gz;
float mx, my, mz;
//Calibration values
uint16_t num_Samples = 100;
float AccelMinX = 50;
float AccelMaxX = -50;
float AccelMinY = 50;
float AccelMaxY = -50;
float AccelMinZ = 50;
float AccelMaxZ = -50;
float accel_temp[3] = {0.0, 0.0, 0.0};
float ax_med, ay_med, az_med;
bool flag_calib = false;
//Tilt estimation values
float thetaA, phiA;
float Xm, Ym;
float theta = 0;
float phi = 0;
float psi = 0;
float dt;
float q0, q1, q2, q3;
unsigned long millisOld;



// an MPU9250 object with the MPU-9250 sensor on I2C bus 0 with address 0x68
MPU9250 IMU(Wire,0x68);
int status;

void setup() {
  // serial to display data
  Serial.begin(9600);
  while(!Serial) {}

  // start communication with IMU 
  status = IMU.begin();
  if (status < 0) {
    Serial.println("IMU initialization unsuccessful");
    Serial.println("Check IMU wiring or try cycling power");
    Serial.print("Status: ");
    Serial.println(status);
    while(1) {}
  }
  // setting the accelerometer full scale range to +/-8G 
  IMU.setAccelRange(MPU9250::ACCEL_RANGE_2G);
  // setting the gyroscope full scale range to +/-500 deg/s
  IMU.setGyroRange(MPU9250::GYRO_RANGE_500DPS);
  // setting DLPF bandwidth to 20 Hz
  IMU.setDlpfBandwidth(MPU9250::DLPF_BANDWIDTH_20HZ);
  // setting SRD to 19 for a 50 Hz update rate
  IMU.setSrd(19);

  millisOld = millis();

  if(flag_calib){
    accel_calibration();
    float offsetX = (AccelMaxX + AccelMinX)/2.0f;
    float offsetY = (AccelMaxY + AccelMinY)/2.0f;
    float offsetZ = (AccelMaxZ + AccelMinZ)/2.0f;
    IMU.setAccelCalX(offsetX, 1.0);
    IMU.setAccelCalY(offsetY, 1.0);
    IMU.setAccelCalZ(offsetZ, 1.0);
    Serial.print("Offset X = "); Serial.print(offsetX);Serial.print(", Offset Y = "); Serial.print(offsetY); Serial.print(", Offset Z = "); Serial.println(offsetZ);
  }
//  Serial.println("Calibrando Magnetometro");
//  IMU.calibrateMag();
//  Serial.println("Calibrado");
//  IMU.setMagCalX(IMU.getMagBiasX_uT(), IMU.getMagScaleFactorX());
//  IMU.setMagCalY(IMU.getMagBiasY_uT(), IMU.getMagScaleFactorY());
//  IMU.setMagCalZ(IMU.getMagBiasZ_uT(), IMU.getMagScaleFactorZ());
  
  //Valores guardados de calibração
  IMU.setAccelCalX(12.86, 1.0f);
  IMU.setAccelCalY(4.68, 1.0f);
  IMU.setAccelCalZ(3.48, 1.0f);
}

void loop() {
  // read the sensor
  IMU.readSensor();
  
  //Get accelerometer measurements
  ax = IMU.getAccelX_mss();
  ay = IMU.getAccelY_mss();
  az = IMU.getAccelZ_mss();
  //Get gyroscope measurements
  gx = IMU.getGyroX_rads();
  gy = IMU.getGyroY_rads();
  gz = IMU.getGyroZ_rads();
  //Get magnetometer measurements
//  mx = IMU.getMagX_uT();
//  my = IMU.getMagY_uT();
//  mz = IMU.getMagZ_uT();
  
  // display the data
  
//  Serial.print("Accel: "); Serial.print(ax); Serial.print(","); Serial.print(ay); Serial.print(","); Serial.println(az);
//  Serial.print("Gyro: "); Serial.print(gx); Serial.print(","); Serial.print(gy); Serial.print(","); Serial.println(gz);
//  Serial.print("Mag: "); Serial.print(mx); Serial.print(","); Serial.print(my); Serial.print(","); Serial.println(mz);

  dt = fusion.deltatUpdate();
//  fusion.MahonyUpdate(gx, gy, gz, ax, ay, az, dt);  //mahony is suggested if there isn't the mag
  fusion.MadgwickUpdate(gx, gy, gz, -ax, -ay, -az, dt);  //else use the magwick
//
//  theta = fusion.getRoll();
//  phi = fusion.getPitch();
//  psi = fusion.getYaw();
  q0 = fusion.getQ0();
  q1 = fusion.getQ1();
  q2 = fusion.getQ2();
  q3 = fusion.getQ3();
  float qnorm = sqrt(q0*q0+q1*q1+q2*q2+q3*q3);
//
//  Serial.print(gx); Serial.print(','); Serial.print(gy); Serial.print(','); Serial.print(gz); Serial.print(','); Serial.print(-ax); Serial.print(','); Serial.print(-ay);
//  Serial.print(','); Serial.print(-az); Serial.print(','); Serial.println(dt);
//  Serial.println("---------------------------------------------------------------------");
  Serial.print(q0); Serial.print(','); Serial.print(q1); Serial.print(',');  Serial.print(q2); Serial.print(','); Serial.print(q3); Serial.print(','); Serial.print(-ax);
  Serial.print(','); Serial.print(-ay); Serial.print(','); Serial.print(-az); Serial.print(','); Serial.print(gx); Serial.print(','); Serial.print(gy); Serial.print(','); Serial.println(gz);
//  Serial.println(qnorm);
//  Serial.println("---------------------------------------------------------------------");
//  Serial.print("Roll:"); Serial.print(phi); Serial.print(","); 
//  Serial.print("Pitch:"); Serial.print(theta); Serial.print(','); 
//  Serial.print("Yaw:"); Serial.println(psi);

  delay(100);
}

void accel_calibration(){
   for(int j=0; j<6; j++){
    
      Serial.print("Set a position... "); Serial.println(j);
      while(!Serial.available()){}
      
      for(int i=0; i<num_Samples; i++){
        IMU.readSensor();
        accel_temp[0] += IMU.getAccelX_mss();
        accel_temp[1] += IMU.getAccelY_mss();
        accel_temp[2] += IMU.getAccelZ_mss();
        delay(20);
      }
    
      Serial.print("Sum:");Serial.print(accel_temp[0]); Serial.print(accel_temp[1]); Serial.println(accel_temp[2]);
      ax_med = accel_temp[0]/num_Samples;
      ay_med = accel_temp[1]/num_Samples;
      az_med = accel_temp[2]/num_Samples;
      accel_temp[0] = 0.0f;
      accel_temp[1] = 0.0f;
      accel_temp[2] = 0.0f;
      Serial.print("Mean:"); Serial.print(ax_med); Serial.print(ay_med); Serial.println(az_med);
      
      if(ax_med < AccelMinX){
        AccelMinX = ax_med;
      }
      if(ax_med > AccelMaxX){
        AccelMaxX = ax_med;
      }
      if(ay_med < AccelMinY){
        AccelMinY = ay_med;
      }
      if(ay_med > AccelMaxY){
        AccelMaxY = ay_med;
      }
      if(az_med < AccelMinZ){
        AccelMinZ = az_med;
      }
      if(az_med > AccelMaxZ){
        AccelMaxZ = az_med;
      }
    
      Serial.print("Accel Minimums: "); Serial.print(AccelMinX); Serial.print("  ");Serial.print(AccelMinY); Serial.print("  "); Serial.print(AccelMinZ); Serial.println();
      Serial.print("Accel Maximums: "); Serial.print(AccelMaxX); Serial.print("  ");Serial.print(AccelMaxY); Serial.print("  "); Serial.print(AccelMaxZ); Serial.println();
     
      while (Serial.available())
      {
        Serial.read();  // clear the input buffer
      }
  }
}

float constrainAngle360(float dta) {
  dta = fmod(dta, 2.0 * PI);
  if (dta < 0.0)
    dta += 2.0 * PI;
  return dta;
}
