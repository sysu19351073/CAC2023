#include <AccelStepper.h>
#include <Servo.h> 
#include <Adafruit_NeoPixel.h>
#include <EEPROM.h>
#include <string.h>

#define PIN 22
#define STEPS_PER_REV 200
#define num 5

int ready = 0;

int enablePin = 8;

//右侧步进电机pin，即xdirpin和xsteppin，为5,2
int motor1DirPin = 5; //digital pin 2
int motor1StepPin = 2; //digital pin 3

//左侧步进电机pin，即zdirpin和zsteppin，为7,4
int motor2DirPin = 7; //digital pin 4
int motor2StepPin = 4; //digital pin 5

//Mux in "SIG" pin“SIG”引脚中的多路复用器
int SIG_pin = 15;
int bluetoothStatusPin = 27;

int motorSpeed = 1000; //maximum steps per second (about 3rps / at 16 microsteps)10000
int motorAccel = 1000; //steps/second/second to accelerate 90000

//定义灯
Adafruit_NeoPixel strip = Adafruit_NeoPixel(9, PIN, NEO_GRB + NEO_KHZ800);

//stepper1右夹爪，stepper2左夹爪
Servo stepper1_armA;  // create servo object to control a servo 
Servo stepper1_armB;
Servo stepper2_armA;
Servo stepper2_armB;

//定义步进电机，全步进运行
AccelStepper stepper1(1, motor1StepPin , motor1DirPin);//set up the accelStepper intances
AccelStepper stepper2(1, motor2StepPin, motor2DirPin);

//连接至舵机的引脚编号A右，B左，1右，2左
int stepper1ArmA = 11; 
int stepper1ArmB = 10;
int stepper2ArmA = 9; 
int stepper2ArmB = 6;

//设定舵机初始转角
int stepper1ArmAHome = 86;  //117增大张开
int stepper1ArmBHome = 88;  //134增大张开
int stepper2ArmAHome = 69.5; //176增大张开
int stepper2ArmBHome = 87.5; //190

//一些特定的角度设定
int stopForce = 540;//current sensor reading at max point
int angle = 30;

//设置标识夹爪状态的参数，初始时夹爪闭合
boolean stepper1Grip = false;
boolean stepper2Grip = false;
boolean stepperEnable = false;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  
  stepper1_armA.attach(stepper1ArmA);
  stepper1_armB.attach(stepper1ArmB);
  stepper2_armA.attach(stepper2ArmA);
  stepper2_armB.attach(stepper2ArmB);

  stepper1_armA.write(stepper1ArmAHome);
  stepper1_armB.write(stepper1ArmBHome);
  stepper2_armA.write(stepper2ArmAHome);
  stepper2_armB.write(stepper2ArmBHome);

  //gripR1();
  //gripL1();

  stepper1Grip = true;
  stepper2Grip = true;

  //设置步进电机引脚反转，反向使能引脚
  // stepper1.setPinsInverted(false,false,true);
  // stepper2.setPinsInverted(false,false,true);

  //设置允许的最大速度，run()函数将加速到该速度
  stepper1.setMaxSpeed(motorSpeed);
  stepper2.setMaxSpeed(motorSpeed);

  //设置加速减速的数率
  stepper1.setAcceleration(motorAccel);
  stepper2.setAcceleration(motorAccel);

  //设置使能引脚，禁用电机引脚输出，即关闭引脚
  // stepper1.setEnablePin(sleep_1);
  // stepper1.disableOutputs();
  // stepper2.setEnablePin(sleep_2);
  // stepper2.disableOutputs();

  //设置当前位置，无论电机现在在什么位置，都被认为是新的0位置
  stepper1.setCurrentPosition(0);
  stepper2.setCurrentPosition(0);

  //灯条初始化
  strip.begin();
  strip.show();

  //设置Arduino引脚为输出(OUTPUT)模式
  pinMode(motor1DirPin, OUTPUT); 
  pinMode(motor1StepPin, OUTPUT); 
  pinMode(motor2DirPin, OUTPUT); 
  pinMode(motor2StepPin, OUTPUT); 
  pinMode(enablePin, OUTPUT);

  //将这些引脚设置为接地？
  digitalWrite(enablePin, LOW);

}

void loop() {
  // put your main code here, to run repeatedly:
 // Fetch all commands that are in the buffer
  //Serial.println("Started");
  stepperEnable=true;
  int trans;
  if(ready==0){
    Serial.println("OK");
  }
  if(ready==2){
    Serial.println("Stop");
  }
  if(Serial.available()){}
  while(1) {
    stepper1.setCurrentPosition(0);
    stepper2.setCurrentPosition(0);
    ready = 1;
    trans = Serial.read();
    // Do something different based on what we got:
//============================================================= Standard Commands 
    //R0
    if(trans=='a'){
      gripR0();
      delay(1000);
      Serial.println("anext");
      break;
    }
    //L0
    if(trans=='A'){
      gripL0();
      Serial.println("Anext");
      break;
    }
    //R1
    if(trans=='b'){
      gripR1();
      Serial.println("bnext");
      break;
    }
    //L1
    if(trans=='B'){
      gripL1();
      Serial.println("Bnext");
      break;
    }
    //R2
    if(trans=='c'){
      twistR2();
      Serial.println("cnext");
      break;
    }
    //L2
    if(trans=='C'){
      twistL2();
      Serial.println("Cnext");
      break;
    }
    //R3
    if(trans=='d'){
      twistR3();
      Serial.println("dnext");
      break;
    }
    if(trans=='D'){
      twistL3();
      Serial.println("Dnext");
      break;
    }
    if(trans=='e'){
      twistR4();
      Serial.println("enext");
      break;
    }
    if(trans=='E'){
      twistL4();
      Serial.println("Enext");
      break;
    }
    if(trans=='f'){
      twistR5();
      Serial.println("fnext");
      break;
    }
    if(trans=='F'){
      twistL5();
      Serial.println("Fnext");
      break;
    }
    if(trans=='x'){
      Serial.println("Stop");
      ready = 2;
      delay(1000);
      //Serial.println("Already finish!");
      break;   
    }
  }
}


//============================================================================================================================================================================================================================================================================================
//============================================================================================================================================================================================================================================================================================
//Twist - used to turn faces of the cube; or trans position of cube
//============================================================================================================================================================================================================================================================================================
//============================================================================================================================================================================================================================================================================================

//true = turning face, false = turning cube
// 逆-，顺+

//顺时针旋转右夹爪90度
void twistR2(){
  stepper1.moveTo(stepper1.currentPosition() + STEPS_PER_REV);
  while(stepper1.distanceToGo()!=0){
    stepper1.run();
  }
  delay(500);
}

//顺时针旋转左夹爪90度
void twistL2(){
  stepper2.moveTo(stepper2.currentPosition() + STEPS_PER_REV);
  while(stepper2.distanceToGo()!=0){
    stepper2.run();
  }
  delay(500);
}

//顺时针旋转右夹爪180度
void twistR3(){
  stepper1.moveTo(stepper1.currentPosition() + 2*STEPS_PER_REV);
  while(stepper1.distanceToGo()!=0){
    stepper1.run();
  }
  delay(500);
}

//顺时针旋转左夹爪180度
void twistL3(){
  stepper2.moveTo(stepper2.currentPosition() + 2*STEPS_PER_REV);
  while(stepper2.distanceToGo()!=0){
    stepper2.run();
  }
  delay(500);
}

//逆时针旋转右夹爪90度
void twistR4(){
  stepper1.moveTo(stepper1.currentPosition() - STEPS_PER_REV);
  while(stepper1.distanceToGo()!=0){
    stepper1.run();
  }
  delay(500);
}

//逆时针旋转左夹爪90度
void twistL4(){
  stepper2.moveTo(stepper2.currentPosition() - STEPS_PER_REV);
  while(stepper2.distanceToGo()!=0){
    stepper2.run();
  }
  delay(500);
}

//逆时针旋转右夹爪180度
void twistR5(){
  stepper1.moveTo(stepper1.currentPosition() - 2*STEPS_PER_REV);
  while(stepper1.distanceToGo()!=0){
    stepper1.run();
  }
  delay(500);
}

//逆时针旋转左夹爪180度
void twistL5(){
  stepper2.moveTo(stepper2.currentPosition() - 2*STEPS_PER_REV);
  while(stepper2.distanceToGo()!=0){
    stepper2.run();
  }
  delay(500);
}

//============================================================================================================================================================================================================================================================================================
//============================================================================================================================================================================================================================================================================================
//Grip - controls the servos responsible for gripping the cubw
//============================================================================================================================================================================================================================================================================================
//============================================================================================================================================================================================================================================================================================
//闭-，开+

//R0操作，张开右侧夹爪
void gripR0(){
  for(int i=0;i<35;i++){
    stepper1_armA.write(stepper1ArmAHome + i);
    stepper1_armB.write(stepper1ArmBHome + i);
    delay(20);
  } 
  stepper1Grip = false; 
  delay(500);
}

//L0操作，张开左侧夹爪
void gripL0(){
    for(int i=0;i<35;i++){
      stepper2_armA.write(stepper2ArmAHome + i);
      stepper2_armB.write(stepper2ArmBHome + i);
      delay(20);
    } 
    stepper2Grip = false;
    delay(500);
}


//R1操作，夹紧右侧夹爪
void gripR1(){
  for(int i=0;i<35;i++){
      stepper1_armA.write(stepper2ArmAHome + 40 - i);
      stepper1_armB.write(stepper2ArmBHome + 39 - i);
      delay(20);
    } 
    stepper1Grip = true;
    delay(500);
}

//L1操作，夹紧左侧夹爪
void gripL1(){
   for(int i=0;i<35;i++){
      stepper2_armA.write(stepper2ArmAHome + 35 - i);
      stepper2_armB.write(stepper2ArmBHome + 35 - i);
      delay(20);
    } 
    stepper2Grip = true;
    delay(500);
}


