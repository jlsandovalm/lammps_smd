/*
 * SmdMatDB.cpp
 *
 *  Created on: Jan 11, 2016
 *      Author: ganzenmueller
 */

#include "SmdMatDB.h"
#include <stdio.h>
#include "SimpleIni.h"
#include "stdlib.h"
#include <iostream>
#include <stdexcept>      // std::out_of_range
#include <Eigen/Eigen>
#include <sstream>

#define SSTR( x ) dynamic_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()
#define DOUBLE_NOT_FOUND -999999999999.0
#define LONG_NOT_FOUND -99999999
#define STRING_NOT_FOUND "STRING_NOT_FOUND"

using namespace std;
using namespace Eigen;

SmdMatDB::SmdMatDB() {
	// TODO Auto-generated constructor stub

	printf("\n\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	printf("\n\nthis is the constructor\n");

}

SmdMatDB::~SmdMatDB() {
	// TODO Auto-generated destructor stub

	printf("\n\n--------------------------------------------------------------------\n");
	printf("\n\nthis is the destructor\n");

	delete[] gProps;
}

int SmdMatDB::ReadMaterials(const int ntypes__) {
	printf("DOING SOMETHING ********************************************\n");
	ntypes = ntypes__;

	CSimpleIni ini;
	ini.Reset();

	// allocate lookup tables for number of lammps types
	printf("allocating storage for %d particle types\n", ntypes);
	gProps = new gProp[ntypes + 1];

	// load from a string
	std::string strData = "materials.ini";
	cout << "reading material database from file: " << strData << endl;
	int rc = ini.LoadFile(strData.c_str());
	if (rc < 0) {
		return -1;
	}

	// we want a general section for each LAMMPS particle type
	for (int i = 1; i < ntypes + 1; i++) {
		int retVal = ReadSectionGeneral(ini, i);
		if (retVal < 0) {
			return -1;
		}
	}

	// read each type's EOS
	for (int i = 1; i < ntypes + 1; i++) {
		// read the EOS for this type
		int retVal = ReadEoss(ini, i);
		if (retVal < 0) {
			return -1;
		}

	}

// read each type's strength model
	for (int i = 1; i < ntypes + 1; i++) {
		// read the strength model for this type
		int retVal = ReadStrengths(ini, i);
		if (retVal < 0) {
			return -1;
		}

	}

	// read each type's viscosity model
	for (int i = 1; i < ntypes + 1; i++) {
		int retVal = ReadViscosities(ini, i);
		if (retVal < 0) {
			return -1;
		}

	}

	DetermineReferenceSoundspeed();

	return 0;
}

int SmdMatDB::ReadSectionGeneral(CSimpleIni &ini, const int itype) {
	int retVal;
	std::map<std::string, int> eosmap, strengthmap;
	eosmap["EOS_LINEAR"] = 1;
	strengthmap["STRENGTH_LINEAR"] = 1;

	string sectionName = SSTR(itype);

	gProps[itype].rho0 = ini.GetDoubleValue(sectionName.c_str(), "rho0", DOUBLE_NOT_FOUND);
	if (gProps[itype].rho0 == DOUBLE_NOT_FOUND) {
		printf("could not read rho0 for type %d\n", itype);
		return -1;
	}

	gProps[itype].cp = ini.GetDoubleValue(sectionName.c_str(), "cp", DOUBLE_NOT_FOUND);
	if (gProps[itype].cp == DOUBLE_NOT_FOUND) {
		printf("could not read cp for type %d\n", itype);
		return -1;
	}

	// read EOS
	string eosName = ini.GetValue(sectionName.c_str(), "EOS", STRING_NOT_FOUND);
	cout << "EOS is " << eosName << endl;
	if (eosName != "NONE") {
		retVal = ini.GetSectionSize(eosName.c_str());
		if (retVal < 0) {
			printf("EOS could not be found\n");
			return -1;
		} else {
			printf("eos could be found\n");
			gProps[itype].eosName = eosName;
		}
	}

	// read STRENGTH
	string strengthName = ini.GetValue(sectionName.c_str(), "STRENGTH", STRING_NOT_FOUND);
	cout << "STRENGTH is " << strengthName << endl;

	if (strengthName != "NONE") {
		retVal = ini.GetSectionSize(strengthName.c_str());
		if (retVal < 0) {
			printf("Strength could not be found\n");
			return -1;
		} else {
			printf("Strength could be found\n");
			gProps[itype].strengthName = strengthName;
		}
	}

	// read Viscosity Model
	string viscName = ini.GetValue(sectionName.c_str(), "VISCOSITY", STRING_NOT_FOUND);
	cout << "VISCOSITY model is " << viscName << endl;

	if (viscName != "NONE") {
		retVal = ini.GetSectionSize(viscName.c_str());

		if (retVal < 0) {
			printf("Viscosity model %s could not be found in ini file\n", viscName.c_str());
			return -1;
		} else {
			printf("Viscosity could be found\n");
			gProps[itype].viscName = viscName;
		}
	}

	return 1;
}

int SmdMatDB::ReadEoss(CSimpleIni &ini, const int itype) {

	// we already know that the section exists
	string section = gProps[itype].eosName;

	if (section != "NONE") {
		int EosId = ini.GetLongValue(section.c_str(), "EosId", LONG_NOT_FOUND);
		if (EosId == LONG_NOT_FOUND) {
			printf("Missing EosId in eos model\n");
			return -1;
		} else {
			if (EosId == 1) {
				// this is the Linear EOS
				return ReadEosLinear(ini, itype);
			} else if (EosId == 2) {
				// this is the Tait-Murnaghan isothermal EOS
				return ReadEosTait(ini, itype);
			} else {
				printf("EosId = %d unknown\n", EosId);
				return -1;
			}
		}
	}

	return 1;

}

int SmdMatDB::ReadEosLinear(CSimpleIni &ini, const int itype) {
	string section = gProps[itype].eosName;

	// check if this material already exists
	for (size_t i = 0; i < eosLinear_vec.size(); ++i) {
		if (eosLinear_vec[i].name == section) {
			gProps[itype].eosType = 1;
			gProps[itype].eosTypeIdx = i;
			return 0;
		}
	}

	double K = ini.GetDoubleValue(section.c_str(), "K", DOUBLE_NOT_FOUND);
	if (K == DOUBLE_NOT_FOUND) {
		printf("could not read K for linear eos model [%s]\n", section.c_str());
		return -1;
	}

	eosLinear_vec.push_back(EosLinear(K, section));
	gProps[itype].eosType = 1;
	gProps[itype].eosTypeIdx = eosLinear_vec.size() - 1;

	return 0;
}

int SmdMatDB::ReadEosTait(CSimpleIni &ini, const int itype) {
	string section = gProps[itype].eosName;

	// check if this material already exists
	for (size_t i = 0; i < eosTait_vec.size(); ++i) {
		if (eosTait_vec[i].name == section) {
			gProps[itype].eosType = 2;
			gProps[itype].eosTypeIdx = i;
			return 0;
		}
	}

	double K = ini.GetDoubleValue(section.c_str(), "K", DOUBLE_NOT_FOUND);
	if (K == DOUBLE_NOT_FOUND) {
		printf("could not read K for Tait eos model [%s]\n", section.c_str());
		return -1;
	}

	int n = ini.GetLongValue(section.c_str(), "n", LONG_NOT_FOUND);
	if (n == LONG_NOT_FOUND) {
		printf("could not read n for Tait eos model [%s]\n", section.c_str());
		return -1;
	}

	eosTait_vec.push_back(EosTait(K, n, section));
	gProps[itype].eosType = 2;
	gProps[itype].eosTypeIdx = eosTait_vec.size() - 1;

	return 0;
}

int SmdMatDB::ReadStrengths(CSimpleIni &ini, const int itype) {

	// we already know that the section exists
	string section = gProps[itype].strengthName;

	if (section != "NONE") {
		int MatId = ini.GetLongValue(section.c_str(), "MatId", LONG_NOT_FOUND);
		if (MatId == LONG_NOT_FOUND) {
			printf("Missing MatId in strength model\n");
			return -1;
		} else {
			if (MatId == 1) {
				// this is the linear strength model
				return ReadStrengthLinear(ini, itype);
			} else if (MatId == 2) {
				// this is the simple plasticity model
				return ReadStrengthSimplePlasticity(ini, itype);
			} else {
				printf("MatId = %d unknown\n", MatId);
				return -1;
			}
		}
	}

	return 1;

}

int SmdMatDB::ReadStrengthLinear(CSimpleIni &ini, const int itype) {
	// initialize new strength model

	string section = gProps[itype].strengthName;

	// check if this material already exists
	for (size_t i = 0; i < strengthLinear_vec.size(); ++i) {
		if (strengthLinear_vec[i].name == section) {
			gProps[itype].strengthType = 1;
			gProps[itype].strengthTypeIdx = i;
			return 0;
		}
	}

	double E = ini.GetDoubleValue(section.c_str(), "E", DOUBLE_NOT_FOUND);
	if (E == DOUBLE_NOT_FOUND) {
		printf("could not read E for linear strength model [%s]\n", section.c_str());
		return -1;
	}

	double nu = ini.GetDoubleValue(section.c_str(), "nu", DOUBLE_NOT_FOUND);
	if (nu == DOUBLE_NOT_FOUND) {
		printf("could not read nu for linear strength model [%s]\n", section.c_str());
		return -1;
	}

	strengthLinear_vec.push_back(StrengthLinear(E, nu, section));
	gProps[itype].strengthType = 1;
	gProps[itype].strengthTypeIdx = strengthLinear_vec.size() - 1;

	return 0;

}

int SmdMatDB::ReadStrengthSimplePlasticity(CSimpleIni &ini, const int itype) {
	string section = gProps[itype].strengthName;

	// check if this material already exists
	for (size_t i = 0; i < strengthSimplePlasticity_vec.size(); ++i) {
		if (strengthSimplePlasticity_vec[i].name == section) {
			gProps[itype].strengthType = 2;
			gProps[itype].strengthTypeIdx = i;
			return 0;
		}
	}

	double E = ini.GetDoubleValue(section.c_str(), "E", DOUBLE_NOT_FOUND);
	if (E == DOUBLE_NOT_FOUND) {
		printf("could not read E for simple plasticity strength model [%s]\n", section.c_str());
		return -1;
	}

	double nu = ini.GetDoubleValue(section.c_str(), "nu", DOUBLE_NOT_FOUND);
	if (nu == DOUBLE_NOT_FOUND) {
		printf("could not read nu for simple plasticity strength model [%s]\n", section.c_str());
		return -1;
	}

	double sigmaYield = ini.GetDoubleValue(section.c_str(), "yield_stress", DOUBLE_NOT_FOUND);
	if (sigmaYield == DOUBLE_NOT_FOUND) {
		printf("could not read sigma_yield for simple plasticity strength model [%s]\n", section.c_str());
		return -1;
	}

	strengthSimplePlasticity_vec.push_back(StrengthSimplePlasticity(E, nu, sigmaYield, section));
	gProps[itype].strengthType = 2;
	gProps[itype].strengthTypeIdx = strengthSimplePlasticity_vec.size() - 1;

	return 0;

}

int SmdMatDB::ReadViscosities(CSimpleIni &ini, const int itype) {

	// we already know that the section exists
	string section = gProps[itype].viscName;
	printf("+++ visc model %s\n", section.c_str());

	if (section != "NONE") {
		int ViscId = ini.GetLongValue(section.c_str(), "ViscId", LONG_NOT_FOUND);
		if (ViscId == LONG_NOT_FOUND) {
			printf("Missing ViscId in visc model\n");
			return -1;
		} else {
			if (ViscId == 1) {
				// this is Newtonian viscosity
				return ReadViscNewton(ini, itype);
			} else {
				printf("ViscId = %d unknown\n", ViscId);
				return -1;
			}
		}
	}

	return 1;

}

int SmdMatDB::ReadViscNewton(CSimpleIni &ini, const int itype) {
	string section = gProps[itype].viscName;

	// check if this material already exists
	for (size_t i = 0; i < viscNewton_vec.size(); ++i) {
		if (viscNewton_vec[i].name == section) {
			gProps[itype].viscType = 1;
			gProps[itype].viscTypeIdx = i;
			printf("returning because viscosity model already exists\n");
			return 0;
		}
	}

	double eta = ini.GetDoubleValue(section.c_str(), "eta", DOUBLE_NOT_FOUND);
	if (eta == DOUBLE_NOT_FOUND) {
		printf("could not read eta for Newton viscosity model [%s]\n", section.c_str());
		return -1;
	} else {
		printf("**** ETA = %f\n", eta);
	}

	viscNewton_vec.push_back(ViscosityNewton(eta, section));
	printf("Assigning visc type %d to particle type %d\n", gProps[itype].viscType, itype);
	gProps[itype].viscType = 1;
	gProps[itype].viscTypeIdx = viscNewton_vec.size() - 1;

	return 0;
}

void SmdMatDB::PrintData() {

	printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");
	printf("... SMD / MPM CONSTITUTIVE MODELS\n\n");

	for (int itype = 1; itype < ntypes + 1; itype++) {

		printf("-------------------------------------------------------------------------------\n");
		printf("nparticle type = %d\n", itype);

		printf("... reference mass density is %f\n", gProps[itype].rho0);
		printf("... reference speed of sound is %f\n", gProps[itype].c0);
		printf("... reference shear modulus is %f\n", gProps[itype].G0);
		printf("\n");

		int eosType = gProps[itype].eosType;
		if (eosType > 0) {
			printf("EOS name is %s\n", gProps[itype].eosName.c_str());
			int idx = gProps[itype].eosTypeIdx;
			if (eosType == 1) { // linear EOS
				eosLinear_vec[idx].PrintParameters();
			} else if (eosType == 2) { // Murnaghan Tait isothermal EOS
				eosTait_vec[idx].PrintParameters();
			}
		} else {
			printf("... no EOS defined\n");
		}

		printf("\n");
		int strengthType = gProps[itype].strengthType;
		if (strengthType > 0) {
			printf("material strength name is %s\n", gProps[itype].strengthName.c_str());
			int idx = gProps[itype].strengthTypeIdx;
			if (strengthType == 1) { // linear elastic
				strengthLinear_vec[idx].PrintParameters();
			} else if (strengthType == 2) { // simple plasticity model
				strengthSimplePlasticity_vec[idx].PrintParameters();
			}
		} else {
			printf("... no material strength model defined\n");
		}

		printf("\n");
		int viscType = gProps[itype].viscType;
		if (viscType > 0) {
			printf("material viscosity name is %s\n", gProps[itype].viscName.c_str());
			int idx = gProps[itype].viscTypeIdx;
			if (viscType == 1) { // Newtonian viscosity
				viscNewton_vec[idx].PrintParameters();
			}
		} else {
			printf("... no viscosity model defined\n");
		}
		printf("-------------------------------------------------------------------------------\n");

	}

	printf("\n>>========>>========>>========>>========>>========>>========>>========>>========\n");

}

void SmdMatDB::ComputePressure(const double mu, const double temperature, const int itype, double &pressure, double &K_eff) {

	int eosType = gProps[itype].eosType;
	int eosIdx = gProps[itype].eosTypeIdx;

	if (eosType == 1) { // linear EOS
		pressure = eosLinear_vec[eosIdx].ComputePressure(mu);
		K_eff = eosLinear_vec[eosIdx].K;
	} else if (eosType == 2) { // Tait Murnaghan isothermal EOS
		eosTait_vec[eosIdx].ComputePressure(mu, pressure, K_eff);
	}
}

void SmdMatDB::ComputeDevStressIncrement(const Matrix3d d_dev, const int itype, const Matrix3d oldStressDeviator,
		double &plasticStrainIncrement, Matrix3d &stressIncrement) {

	int strengthType = gProps[itype].strengthType;
	int strengthIdx = gProps[itype].strengthTypeIdx;

	if (strengthType == 1) { // linear strength model
		stressIncrement = strengthLinear_vec[strengthIdx].ComputeStressIncrement(d_dev);
	} else if (strengthType == 2) { // simple plasticity strength model
		strengthSimplePlasticity_vec[strengthIdx].ComputeStressIncrement(d_dev, oldStressDeviator, plasticStrainIncrement,
				stressIncrement);
	} else {
		stressIncrement.setZero();
	}
}

void SmdMatDB::ComputeViscousStress(const Eigen::Matrix3d d_dev, const int itype, Eigen::Matrix3d &viscousStress) {
	int viscType = gProps[itype].viscType;
	int viscIdx = gProps[itype].viscTypeIdx;

	if (viscType == 1) { // Newton viscosity
		viscousStress = viscNewton_vec[viscIdx].ComputeStressDeviator(d_dev);
	}

}

void SmdMatDB::DetermineReferenceSoundspeed() {

	for (int itype = 1; itype < ntypes + 1; itype++) {
		int eosType = gProps[itype].eosType;
		if (eosType > 0) {
			int idx = gProps[itype].eosTypeIdx;
			if (eosType == 1) { // linear EOS
				gProps[itype].K0 = eosLinear_vec[idx].K;
			} else if (eosType == 2) { // Murnaghan Tait isothermal EOS
				gProps[itype].K0 = eosTait_vec[idx].K;
			}

		} else {
			printf("\n\n *** WARNING ***\n Could not define reference sped of sound because no EOS is defined\n *** WARNING ***\n");
		}

		gProps[itype].c0 = sqrt(gProps[itype].K0 / gProps[itype].rho0);

		int strengthType = gProps[itype].strengthType;
		if (strengthType > 0) {
			int idx = gProps[itype].strengthTypeIdx;
			if (strengthType == 1) { // linear elastic
				gProps[itype].G0 = strengthLinear_vec[idx].G;
			} else if (strengthType == 2) { // simple plasticity model
				gProps[itype].G0 = strengthSimplePlasticity_vec[idx].G;
			}
		}

	}

}

