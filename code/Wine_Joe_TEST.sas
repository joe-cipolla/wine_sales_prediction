

%let PATH 		= /home/josephcipolla20180/sasuser.v94/UNIT03/HW;
%let NAME 		= P411;
%let LIB 		= &NAME..;
%let INFILE 	= &LIB.WINE_TEST;

%let TEMPFILE 	= TEMPFILE;
%let FIXFILE	= FIXFILE;
%let VARLIST	= VARLIST;


libname &NAME. "&PATH.";


proc print data=&INFILE.(obs=10);
run;



data &TEMPFILE.;
set &INFILE.;
TARGET_FLAG = ( TARGET > 0 );
TARGET_AMT = TARGET - 1;
if TARGET_FLAG = 0 then TARGET_AMT = .;

IMP_STARS				= STARS;
IMP_Density				= Density;
IMP_Sulphates			= Sulphates;
IMP_Alcohol				= Alcohol;
IMP_LabelAppeal			= LabelAppeal;
IMP_TotalSulfurDioxide	= TotalSulfurDioxide;
IMP_ResidualSugar		= ResidualSugar;
IMP_Chlorides			= Chlorides;
IMP_FreeSulfurDioxide	= FreeSulfurDioxide;
IMP_pH					= pH;
M_STARS					= 0;
M_Density				= 0;
M_Sulphates				= 0;
M_Alcohol				= 0;
M_LabelAppeal			= 0;
M_TotalSulfurDioxide	= 0;
M_ResidualSugar			= 0;
M_Chlorides				= 0;
M_FreeSulfurDioxide		= 0;
M_pH					= 0;

if missing(STARS)				then do;	IMP_STARS				= 2;			M_STARS 				= 1; 	end;
if missing(Density)				then do; 	IMP_Density 			= 0.9942027; 	M_Density 				= 1;	end;
if missing(Sulphates)			then do; 	IMP_Sulphates			= 0.5271118; 	M_Sulphates 			= 1;	end;
if missing(Alcohol)				then do; 	IMP_Alcohol				= 10.4892363; 	M_Alcohol 				= 1;	end;
if missing(LabelAppeal)			then do; 	IMP_LabelAppeal			= -0.009066; 	M_LabelAppeal	 		= 1;	end;
if missing(TotalSulfurDioxide)	then do; 	IMP_TotalSulfurDioxide	= 120.7142326;  M_TotalSulfurDioxide 	= 1;	end;
if missing(ResidualSugar)		then do; 	IMP_ResidualSugar		= 5.4187331; 	M_ResidualSugar	 		= 1;	end;
if missing(Chlorides)			then do; 	IMP_Chlorides			= 0.0548225; 	M_Chlorides 			= 1;	end;
if missing(FreeSulfurDioxide)	then do; 	IMP_FreeSulfurDioxide 	= 30.8455713; 	M_FreeSulfurDioxide 	= 1;	end;
if missing(pH)					then do; 	IMP_pH				 	= 3.2076282; 	M_pH					= 1;	end;

*IMP_TotalSulfurDioxide = sign( IMP_TotalSulfurDioxide ) * sqrt( abs(IMP_TotalSulfurDioxide)+1 );
*IMP_TotalSulfurDioxide = sign( IMP_TotalSulfurDioxide ) * log( abs(IMP_TotalSulfurDioxide)+1 );

/* if IMP_TotalSulfurDioxide	< -330 then IMP_TotalSulfurDioxide = -330; */
/* if IMP_TotalSulfurDioxide	> 630  then IMP_TotalSulfurDioxide = 630; */


Norm_FixedAcidity 		= 	(FixedAcidity - 7.0757171) / 6.3176435;
Norm_VolatileAcidity 	= 	(VolatileAcidity - 0.3241039) / 0.7840142;
Norm_CitricAcid 		= 	(CitricAcid - 0.3084127) / 0.8620798;
Norm_ResidualSugar 		= 	(IMP_ResidualSugar - 5.4187331) / 33.7493790;
Norm_Chlorides 			= 	(IMP_Chlorides - 0.0548225) / 0.3184673;
Norm_FreeSulfurDioxide 	= 	(IMP_FreeSulfurDioxide - 30.8455713) / 148.7145577;
Norm_TotalSulfurDioxide = 	(IMP_TotalSulfurDioxide - 120.7142326) / 231.9132105;
Norm_Density 			= 	(Density - 0.9942027) / 0.0265376;
Norm_pH 				= 	(IMP_pH - 3.2076282) / 0.6796871;
Norm_Sulphates			= 	(IMP_Sulphates - 0.5271118) / 0.9321293;
Norm_Alcohol			= 	(IMP_Alcohol - 10.4892363) / 3.7278190;
Norm_AcidIndex			= 	(AcidIndex - 7.7727237) / 1.3239264;



TEST_FLAG = ranuni(3) < 0.2;



keep	
		INDEX
		TARGET
		TARGET_FLAG
		TARGET_AMT
		FixedAcidity
		VolatileAcidity
		CitricAcid
		Density
		LabelAppeal
		AcidIndex
		IMP_STARS
		IMP_Density
		IMP_Sulphates
		IMP_Alcohol
		IMP_LabelAppeal
		IMP_TotalSulfurDioxide
		IMP_ResidualSugar
		IMP_Chlorides
		IMP_FreeSulfurDioxide
		IMP_pH
		M_STARS
		M_Density
		M_Sulphates
		M_Alcohol
		M_LabelAppeal
		M_TotalSulfurDioxide
		M_ResidualSugar
		M_Chlorides
		M_FreeSulfurDioxide
		M_pH
		Norm_FixedAcidity
		Norm_VolatileAcidity
		Norm_CitricAcid
		Norm_ResidualSugar
		Norm_Chlorides
		Norm_FreeSulfurDioxide
		Norm_TotalSulfurDioxide
		Norm_Density
		Norm_pH
		Norm_Sulphates
		Norm_Alcohol
		Norm_AcidIndex
		;

run;



data &FIXFILE.;
set &TEMPFILE.;
run;



data SCOREFILE;
set &FIXFILE.;


P_REGRESSION =	
											(4.41175)		+
				VolatileAcidity			*	(-0.09649)		+
				Density					*	(-0.80653)		+
				LabelAppeal				*	(0.46643)		+
				AcidIndex				*	(-0.19991)		+
				IMP_STARS				*	(0.77939)		+
				IMP_Sulphates			*	(-0.03112)		+
				IMP_Alcohol				*	(0.01240)		+
				IMP_Chlorides			*	(-0.11737)		+
				IMP_FreeSulfurDioxide	*	(0.00028507)	+
				IMP_pH					*	(-0.03153)		+
				M_STARS					*	(-2.24420)		+
				Norm_TotalSulfurDioxide	*	(0.05206)
				;

P_GENMOD_NB = 	
											(1.7858)		+
				VolatileAcidity			*	(-0.0311)		+
				Density					*	(-0.2800)		+
				LabelAppeal				*	(0.1590)		+
				AcidIndex				*	(-0.0805)		+
				IMP_STARS				*	(0.1878)		+
				IMP_Sulphates			*	(-0.0118)		+
				IMP_Alcohol				*	(0.0035)		+
				IMP_Chlorides			*	(-0.0369)		+
				IMP_FreeSulfurDioxide	*	(0.0001)		+
				IMP_pH					*	(-0.0129)		+
				M_STARS					*	(-1.0234)		+
				Norm_TotalSulfurDioxide	*	(0.0186)
				;
P_GENMOD_NB = exp(P_GENMOD_NB);



P_LOGIT_PROB = 
											(3.3739)		+
				VolatileAcidity			*	(-0.1825)		+
				Density					*	(-0.6139)		+
				LabelAppeal				*	(-0.4690)		+
				AcidIndex				*	(-0.3882)		+
				IMP_STARS				*	(2.5601)		+
				IMP_Sulphates			*	(-0.1087)		+
				IMP_Alcohol				*	(-0.0211)		+
				IMP_Chlorides			*	(-0.1551)		+
				IMP_FreeSulfurDioxide	*	(0.000611)		+
				IMP_pH					*	(-0.1836)		+
				M_STARS					*	(-4.3751)		+
				Norm_TotalSulfurDioxide	*	(0.1973)	
				;
if P_LOGIT_PROB > 1000 then P_LOGIT_PROB = 1000;
if P_LOGIT_PROB < -1000 then P_LOGIT_PROB = -1000;
P_LOGIT_PROB = exp(P_LOGIT_PROB) / (1+exp(P_LOGIT_PROB));



P_GENMOD_HURDLE = 	
											(1.1650)		+
				VolatileAcidity			*	(-0.0129)		+
				Density					*	(-0.3669)		+
				LabelAppeal				*	(0.2954)		+
				AcidIndex				*	(-0.0202)		+
				IMP_STARS				*	(0.1209)		+
				IMP_Sulphates			*	(0.0002)		+
				IMP_Alcohol				*	(0.0090)		+
				IMP_Chlorides			*	(-0.0226)		+
				IMP_FreeSulfurDioxide	*	(0.0000)		+
				IMP_pH					*	(0.0097)		+
				M_STARS					*	(-0.2069)		+
				Norm_TotalSulfurDioxide	*	(-0.0075)
				;
P_GENMOD_HURDLE = exp(P_GENMOD_HURDLE);


P_HURDLE = P_LOGIT_PROB * (P_GENMOD_HURDLE+1);


P_ENSEMBLE = (P_REGRESSION + P_GENMOD_NB + P_HURDLE)/3;


P_REGRESSION 	= round(P_REGRESSION	, 1);
P_GENMOD_NB 	= round(P_GENMOD_NB		, 1);
P_HURDLE 		= round(P_HURDLE		, 1);
P_ENSEMBLE		= round(P_ENSEMBLE		, 1);
/* P_GENMOD_ZINB 	= round(P_GENMOD_ZINB	, 1); */

run;

/*  */
/* proc print data=SCOREFILE(obs=25); */
/* var P_ZERO_PROB X_GENMOD_PZERO ; */
/* run; */
/*  */
/* proc print data=SCOREFILE(obs=25); */
/* var P_GENMOD_ZINB X_GENMOD_ZINB P_ZERO_PROB; */
/* run; */


data SCOREFILE2;
set SCOREFILE;

if TEST_FLAG = 0 then delete;

	E_REGRESSION 	= abs(TARGET - P_REGRESSION);
	E_GENMOD_NB 	= abs(TARGET - P_GENMOD_NB);
	E_HURDLE 		= abs(TARGET - P_GENMOD_HURDLE);
/* 	E_GENMOD_ZINB 	= abs(TARGET - P_GENMOD_ZINB); */
	E_ENSEMBLE		= abs(TARGET - P_ENSEMBLE);
run;

proc means data=SCOREFILE2 mean sum min max std;
var E_REGRESSION E_GENMOD_NB E_HURDLE E_ENSEMBLE ;
run;

proc print data=SCOREFILE;
var INDEX P_ENSEMBLE;
run;












