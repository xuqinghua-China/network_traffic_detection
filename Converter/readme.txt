Follow the following steps to convert pcap file to image

1. Run 1_Pcap2Session.ps1 
  powershell 1_Pcap2Session.ps1 -f
  
2. Run powershell 2_ProcessSession.ps1 -a -s

3. python3 3_Session2png.py

4. python3 4_Png2Mnist.py
