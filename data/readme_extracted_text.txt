--- PAGE 1 ---
  1 
Documentation for Mill Data Set 
 
Kai Goebel (NASA Ames) & Alice Agogino (UC Berkeley) 
 
The data in this set represents experiments from ru ns on a milling machine under various operating 
conditions. In particular, tool wear was investigat ed (Goebel, 1996) in a regular cut as well as entry  cut 
and exit cut. Data sampled by three different types  of sensors (acoustic emission sensor, vibration 
sensor, current sensor) were acquired at several positions. 
 
The data is organized in a 1x167 matlab struct array with fields as shown in Table 1 below: 
 
Table 1: Struct field names and description 
Field name Description 
    case Case number (1-16) 
    run Counter for experimental runs in each case 
    VB Flank wear, measured after runs; Measurement s for VB were not taken after each run 
    time Duration of experiment (restarts for each case) 
    DOC Depth of cut (does not vary for each case) 
    feed Feed (does not vary for each case) 
    material Material (does not vary for each case)  
    smcAC AC spindle motor current 
    smcDC DC spindle motor current 
    vib_table Table vibration 
    vib_spindle Spindle vibration 
    AE_table Acoustic emission at table 
    AE_spindle Acoustic emission at spindle 
 
There are 16 cases with varying number of runs. The  number of runs was dependent on the degree of 
flank wear that was measured between runs at irregu lar intervals up to a wear limit (and sometimes 
beyond). Flank wear was not always measured and at times when no measurements were taken, no entry 
was made. 
 
The 16 cases are enumerated in Table 2 
 
Table 2: Experimental conditions 
Case Depth of Cut Feed Material 
1 1.5 0.5 1 – cast iron 
2 0.75 0.5 1 – case iron 
3 0.75 0.25 1 – cast iron 
4 1.5 0.25 1 – cast iron 
5 1.5 0.5 2 – steel 
6 1.5 0.25 2 – steel 
7 0.75 0.25 2 – steel 
8 0.75 0.5 2 – steel 
9 1.5 0.5 1 – cast iron 
10 1.5 0.25 1 – cast iron 
11 0.75 0.25 1 – cast iron 
12 0.75 0.5 1 – cast iron 
13 0.75 0.25 2 – steel --- PAGE 2 ---
  2 
14 0.75 0.5 2 – steel 
15 1.5 0.25 2 – steel 
16 1.5 0.5 2 – steel 
 
 
Experimental Setup 
The setup of the experiment is as depicted in Figure 1 below.  
 
CHARGE AMPLIFIER
CHARGE AMPLIFIER
ACOUSTIC EMISSION 
SENSOR SPINDLE
ACOUSTIC EMISSION 
SENSOR TABLE
VIBRATION SENSOR 
SPINDLE
VIBRATION SENSOR 
TABLE
SPINDLE MOTOR 
CURRENT SENSOR
PREAMPLIFIER RMS
RMSPREAMPLIFIER
LP/HP FILTER RMS
LP/HP FILTER RMS
COMPUTER
RECORDER
 
Figure 1 - Experimental Setup 
 
 
The basic setup encompasses the spindle and the tab le of the Matsuura machining center MC-510V. An 
acoustic emission sensor and a vibration sensor are  each mounted to the table and the spindle of the 
machining center. The signals from all sensors are amplified and filtered, then fed through two RMS 
before they enter the computer for data acquisition. The signal from a spindle motor current sensor is fed 
into the computer without further processing. 
 
The matrix for the parameters chosen for the experi ments were guided by industrial applicability and 
recommended manufacturer’s settings. Therefore, the  cutting speed was set to 200 m/min which is 
equivalent to 826 rev/min. Two different depths of cut were chosen, 1.5mm and 0.75mm. Also, two 
feeds were taken, 0.5mm/rev and 0.25mm/rev which tr anslate into 413mm/min and 206.5mm/min, 
respectively. Two types of material, cast iron and stainless steel J45 were used and, as already 
mentioned earlier, with an inserts of type KC710. T hese choices equal 8 different settings. All 
experiments were done a second time with the same p arameters with a second set of inserts. The size of  
the workpieces was 483mm x 178mm x 51mm. 
 
Data Acquisition and Processing 
As described in the previous section, the data were sent through a high speed data acquisition board with 
maximal sampling rate of 100 KHz. The sampled outpu t of the data was used for the signal processing 
software. LabVIEW 
® (National Instruments, USA) was used for this task . This software is a general 
purpose programming development system which uses a  graphical language (G). With G, programs are --- PAGE 3 ---
  3 
created in block diagram form. The chosen layout al lowed for data acquisition, storage, presentation, 
and processing. Data were stored to allow for real time simulation and also later analysis.  
 
Several sensor signals underwent preprocessing. In most cases, the signal was amplified to be able to 
meet threshold requirements of equipment. In partic ular, the signals from the acoustic emission sensor s 
and from the vibration sensors were amplified to be  in the range of ±5V for maximum load, considering 
the maximum allowable range of the equipment. The s ignals were filtered by a high pass filter, the 
vibration sensor signals were additionally filtered with a low pass filter. Corner frequencies were ch osen 
according to the noise that could be observed on an  oscilloscope. Periodical noise of 180Hz was 
observed on the oscilloscope for the vibration sign al corresponding to the third harmonic of the main 
power supply. Therefore, the chosen corner frequenc y for the low pass filter was 400Hz. For the high 
pass filter, 1kHz was chosen. Above 8KHz, the range  of the acoustic emission sensor ends. That is, 
readings above that frequency cannot be attributed to any occurrence in the machining process. Since i t 
clutters the signal unnecessarily, it was filtered out. Acoustic emission and vibration signals were f ed 
through an RMS device. Its use smoothes the signal and makes it more accessible to signal processing. 
The RMS is proportional to the energy contents of the signal, according to the formula:  
  
RMS = 1
∆T
f
2
t( )dt
0
∆T
∫  
where: 
 ∆T = time constant 
 f(t) = signal function 
The sampling rate has to be smaller than the time c onstant to ensure proper data sampling. The chosen 
parameters were: 
 ∆T = 8.00ms 
 sampling rate: 250Hz 
 
Apart from the preprocessed data, raw data of the a coustic emission of the table and the vibration of the 
table were recorded on a tape recorder to allow for  future comparison and evaluation. It might be of 
interest for feature extraction to have data that did not undergo previous selection; in the same spirit, it is 
worthwhile to be able to use data that are not “cor rupted” yet for neural network techniques or data 
clustering algorithms. Typical sensor readings from  spindle motor current AC and DC portion, acoustic 
emission at the table, vibration at the table, acou stic emission at the spindle, and vibration at the spindle 
are displayed in Figure 7 - Figure 12  in the appendix 
 
Detailed Setup Description 
Mounted to the table is an acoustic emission sensor  model WD 925 (PHYSICAL ACOUSTIC GROUP, 
frequency range up to 2MHz). The acoustic emission sensor is glued to a custom made base which in 
turn is attached to the clamping support. The layou t of the sensors on the clamping device is shown in  
Figure 2. 
 --- PAGE 4 ---
  4 
 
Figure 2 - Clamping device with mounted sensors 
 
The signal from the acoustic emission sensor goes i nto the single ended terminal of an acoustic emission 
preamplifier (DUNEGAN/ENDEVCO, model 1801 with inte grated 50KHZ high pass filter). The signal 
is then amplified by a dual amplifier DE model 302A  (DUNEGAN/ENDEVCO). The signal is then fed 
into the RMS meter which is a custom made device bu ilt by the LMA of the University of California at 
Berkeley. The time constant is set to 8.0ms. The si gnal is then fed into the PHOENIX CONTACT 
UMK-SE 11,25 cable which feeds the signal into a MI O-16 high speed data acquisition board (National 
Instruments). The data acquisition board is mounted in a IBM PC 486DX/2-66. 
 
Also mounted on the clamping device on the table is  a vibration sensor, an accelerometer (model 7201-
50, ENDEVCO) with a frequency range up to 13KHz. It s signal is fed into an ENDEVCO 104 charge 
amplifier with sensitivity 5.71 and 100mV/g output. The signal is then fed into an ITHACO 4302 DUAL 
24dB/octave filter with corner frequencies 400 Hz and 1KHz. Following is a RMS meter same make and 
settings as described earlier. The signal is then f ed then through the PHOENIX CONTACT UMK-SE 
11,25 cable connector into the high speed data acquisition board of the computer. 
 
The same vibration sensor that is mounted on the ta ble is also attached to the spindle into a preexist ing 
threaded hole close to the tool. The signal follows  the same path as described for the other vibration  
sensor except that the signal is fed into the PHOENIX CONTACT cable connector. 
 
Signals from another acoustic emission sensor mounted into another threaded hole on the spindle next to 
the tool is fed into the differential terminal of a n acoustic emission preamplifier model 1801 
(DUNEGAN/ENDEVCO) and then follows the same path as  outlined for the other acoustic emission 
sensor. 
 
A OMRON K3TB-A1015 current converter, powered by a HP 6237B triple output power supply 
providing 15V, feeds the signal from one spindle motor current phase into the cable connector. 
 
A model CTA 213 current sensor (Flexcore Div. of Ma rlan & Associates, Inc.) which uses the same 
phase of the spindle motor current is fed into the cable connector. 
 
Terminals 33 (ground) and 38 (+5V) are used with a switch to trigger the data acquisition. 
 --- PAGE 5 ---
  5 
With industrial applicability in mind, a 70mm face mill with 6 inserts (Figure 3) was chosen as the to ol. 
The inserts KC710 was selected based on the recomme ndations for roughing (Kennametal, 1985). 
KC710 is coated with multiple layers of titanium ca rbide, titanium carbonitride, and titanium nitride 
(TiC/TiC-N/TiN) in sequence. These layers retain th e toughness of tungsten carbide but have improved 
resistance to cratering and edge wear. At the same time, they have the advantage of titanium carbide 
plus reduced face friction. This insert is recommended for heavy roughing.  
 
 
Figure 3 - Schematic of tool and inserts of face mill 
 
The Cutting Process 
The interaction of work piece and tool is rather complex and creates a manifold of physical effects, some 
of which can be captured with sensors. During the e ngagement of the tool in the work piece, plastic 
deformation takes place in the shear zone and a chi p is formed. Energy is released which results in th e 
radiation of heat, cutting forces, vibration and ac oustic emission. Sensors will be used to capture th e 
latter three; they will be explained in more detail. 
 
insert  
primary shear zone  
secondary shear zone  
insert  
workpiece  
workpiece  
FRONT VIEW  SIDE VIEW  
chip  
direction of cut  direction  
of cut  
 
Figure 4 - Shear zones at tool/workpiece interface 
 
Acoustic emission is a high frequency oscillation w hich occurs spontaneously within metals when they 
are deformed or fractured. It is caused by the rele ase of strain energy as the micro structure of the 
material is rearranged. Acoustic emission is genera ted in the shear zones (Figure 4), the primary as w ell --- PAGE 6 ---
  6 
as the secondary along the chip/tool interface thro ugh bulk deformation and sliding, respectively, and , 
lastly, at the tool flank/workpiece interface due t o friction (Schey, 1977).  Although the basic signal of 
the acoustic emission is sinusoidal, it acquires a random pattern due to reflection and scattering due  to 
structural defects. The range of the oscillation li es between 50kHz and several MHz. Because the signa l 
becomes weaker with increasing distance from the cu tting zone, it is desirable to place an acoustic 
emission sensor close to the cutting process which produces problems of properly protecting the sensor. 
Vibration emission is a low frequency oscillation d ue to the acceleration of the object because of the  
dynamic changes of cutting forces resulting from pe riodical changes in tool geometry, chip formation, 
and built up edges. The range of the signal is 0-40 kHz. As with acoustic emission, the vibration senso r 
should be placed close to the cutting zone because the signal gets weaker with increasing distance fro m 
its source. Because the Matsuura machining center i s very rigid, vibrations will be significantly lowe r 
than in the experiments made on the upright Bridgep ort milling machine. However, they still contribute  
to tool wear and hence have to be taken into consideration. 
 
Cutting forces appear in the primary and secondary shear zone where plastic deformation takes place. 
Friction between chip and tool and work piece and t ool also contribute to cutting forces. Sensors 
measuring the cutting forces can capture the force directly and indirectly. The direct method requires the 
installation of force sensors under the work piece which can be cumbersome and costly. The indirect 
method can measure the deflection of machine parts,  motor current of spindle motor or feed motors, and  
power consumption of spindle or feed motors. Since the spindle motor current is proportional to the 
torque which in turn is proportional to the cutting  force, using this signal is an easy way to capture  the 
force signal. Although perhaps not as accurate as t he direct measurement, it is a relatively cheap way  
and the sensor is easy to install. Using the feed m otor current gives information about the feed force  
which is duplicated somewhat by the spindle motor c urrent signal. Likewise, power consumption is also 
directly related to spindle motor current. 
 
Tool Wear  
A high quality product often implies high quality s urface finish and dimensional accuracy. Ideally, a 
sharp tool should be maintained at all times. A dul l one deforms the surface to a greater depth and ma y 
tear the surface which in turn may lower the fatigu e resistance. A worn tool also results in more fric tion 
which in turn results in higher cutting temperature s. Unwanted effects may arise from these 
temperatures, e.g. it may produce untempered marten site in heat treatable steel (Schey, 1977). 
Therefore, tool wear has to be controlled. 
 
Tool wear comes in different forms. Apart from the intuitive rounding of the cutting edge, crater wear  
on the rake face due to the abrasion of the sliding  of the chip on the rake face and flank wear due to  
friction of the tool on the workpiece occur. Speed of cutting, more than other parameters, influence t he 
rate of wear; depth of cut and feed rate also affec t the tool life. In our experiments, we measured th e 
flank wear VB as a generally accepted parameter for evaluating tool wear (Figure 5). 
 --- PAGE 7 ---
  7 
VB
flank face of insert
abrasive wear
 
Figure 5 - Tool wear VB as it is seen on the insert 
 
 
The flank wear VB is measured as the distance from the cutting edge to the end of the abrasive wear on  
the flank face of the tool. The flank wear was obse rved during the experiments. The insert was taken o ut 
of the tool and the wear was measured with the help  of a microscope. The results of one of these 
experiments is displayed in Figure 6. 
 
 
0 
0.1  
0.2  
0.3  
0.4  
0.5  
0.6  
0 10  20  30  40  50  60  70  80  90  
VB [mm]  
machining time [mm]  
material: cast iron  
speed: 826 rev/min  
feed: 206.5 mm/min  
depth of cut: 0.75mm  
 
Figure 6 - Tool wear VB over time 
 
 
Lastly, chipping and fracture are other forms of to ol wear caused by discontinuous machining, 
inclusions in the material, and overloading the too l (Schey, 1977). Mathematical functions for gradual  
wear try to describe tool wear, for example Taylor’s formula: 
T = C vv c
k
 
 where: 
  T = tool life 
  v c = cutting speed 
  k = Taylor exponent --- PAGE 8 ---
  8 
  C v = constant related to 1 min tool life 
 
k and C v are characteristics of the tool and the workpiece material combination. The limitations of this 
approach are that different materials and wear due to the influence of feed cannot be accounted for. 
Apart from that the problem with tool wear in gener al is that it may not be very predictive; slight 
variations in a seemingly same setting vary the too l life considerably. These variations are manifold.  
One is that the same material can have different st rengths which is often the case with cast iron. But  this 
variation can also be local in form of inclusions. These can initiate increased wear when they remove 
large chunks of an insert. Even is the abrasion is of a smaller degree, the geometry of the insert cha nges, 
i.e., the insert becomes less sharp and as a result  the tool has to plow through the material with gre ater 
force, more friction and, as a result, greater tool wear. Other variations are the amount of coolant used to 
control the temperature of the cutting process. Two  thirds of the heat generated during the cutting 
process is removed via chip and tool. Changing this  temperature may have an effect on chemical 
processes by allowing alloying elements to diffuse into the workpiece and thus weaken the structure of  
the insert. Furthermore, the depth of cut may not b e steady, in particular when machining a new rough 
surface for the first time. All of these influences  may be small, but they can add up over the period of 
machining to cause considerable uncertainty in predicting a priori the tool life. 
The use of additional parameters s i in Taylor’s formula to result in 
T = C vv c
k
s
i
 
try to incorporate such influences, however only wi th modest success. These shortcomings again 
motivate the use of non-mathematical techniques. 
 
Appendix 
 
The appendix shows some typical signatures from the different sensors 
 
 
Figure 7 - Spindle motor current (AC) 
 
 
--- PAGE 9 ---
  9 
 
Figure 8 - Spindle motor current (DC portion) 
 
 
Figure 9 -Acoustic emission (table)  
 
 
 
Figure 10 - Acoustic emission (spindle)  
 
 
 
Figure 11 - Vibration (table)  
 --- PAGE 10 ---
  10 
 
Figure 12 - Vibration (spindle)  
 
 
References 
 
K. Goebel, Management of Uncertainty in Sensor Vali dation, Sensor Fusion, and 
Diagnosis of Mechanical Systems Using Soft Computin g Techniques, Ph.D. 
Thesis, Department of Mechanical Engineering, Unive rsity of California at 
Berkeley, 1996. 