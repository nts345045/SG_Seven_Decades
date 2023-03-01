Processing Scripts for "Multi-decadal enhancement of sliding at Saskatchewan Glacier"

This repository contains scripts to process data presented in Stevens and others (202X)

Directory structure and purpose

/root
	/DATA
		/fetch_data.sh - this script pulls data into /RAW from the MINDS@UW repository
		/RAW - holding directory for minimially processed observations archived at MINDS@UW
			/Weather_Ablation
			/GPS
			/SEISMIC
			/VISCOSITY
		/INTERMEDIATE - This directory holds intermediate data files generated during processing
			/GPS
			/SEISMIC
			/VISCOSITY
	/Runoff_Supply_Processing
		/Calculate_mf.py
		/Process_Rdot_surf.py
	/GPS_Processing
		/Continuous
			/Step1_Cleaning.py
			/Step2_Rotation.py
			/Step3_Velocity.py
		/Campaign
			/Velocity_Vector_Processing.py
	/Deformation_Modeling
		/Viscosity_Inversion.py
		/Vint_Calculation.py
	/Seismic_Processing
		/Active
			/TTvO_Processing.py
		/Passive
			/MSEED2ASCII.py
			/Construct_OpenHVSR_Project.py
			/Extract_OpenHVSR_Results.py
	/Figure_Generation
		/Figure_1
		/Figure_2
		/Figure_3
		/Figure_4
		/Figure_5
		/Figure_6
		/Figure_7
		/Figure_S1
		/Figure_S2
		/Figure_S3
		/Figure_S4
		/Figure_S5
		/Figure_S6
		/Figure_S7