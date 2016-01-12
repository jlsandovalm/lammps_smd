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

//	// get all sections
//	CSimpleIniA::TNamesDepend sections;
//	ini.GetAllSections(sections);

// output all of the items
//	CSimpleIniA::TNamesDepend::const_iterator i;
//	for (i = sections.begin(); i != sections.end(); ++i) {
//		printf("section-name = '%s'\n", i->pItem);
//
//		// check if section name is known and if so, call a specific function to read its values
//
//		if (strcmp(i->pItem, "ASSIGN") == 0) {
//			printf("common\n");
//		}
//	}

//	// get the size of the section [standard]
//	printf("\n-- Number of keys in section [standard] = %d\n", ini.GetSectionSize("SECTION"));
//	// get the value of a key
//	int ivalue = ini.GetLongValue("SECTION", "hurz");
//
//	double dvalue = ini.GetDoubleValue("SECTION", "blurb");
//	cout << "value of hurz: " << ivalue << endl;
//	cout << "value of blurb: " << dvalue << endl;

//PrintData();

	return 0;
}

int SmdMatDB::ReadSectionGeneral(CSimpleIni &ini, const int itype) {
	int retVal;
	std::map<std::string, int> eosmap, strengthmap;
	eosmap["EOS_LINEAR"] = 1;
	strengthmap["STRENGTH_LINEAR"] = 1;

	string sectionName = SSTR(itype);
	gProps[itype].c0 = ini.GetDoubleValue(sectionName.c_str(), "c0", DOUBLE_NOT_FOUND);
	if (gProps[itype].c0 == DOUBLE_NOT_FOUND) {
		printf("could not read c0 for type %d\n", itype);
		return -1;
	}

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
	retVal = ini.GetSectionSize(eosName.c_str());
	if (retVal < 0) {
		printf("EOS could not be found\n");
		return -1;
	} else {
		printf("eos could be found\n");
		gProps[itype].eosName = eosName;
	}

	// read STRENGTH
	string strengthName = ini.GetValue(sectionName.c_str(), "STRENGTH", STRING_NOT_FOUND);
	cout << "STRENGTH is " << strengthName << endl;

	retVal = ini.GetSectionSize(strengthName.c_str());
	if (retVal < 0) {
		printf("Strength could not be found\n");
		return -1;
	} else {
		printf("Strength could be found\n");
		gProps[itype].strengthName = strengthName;
	}

	return 1;
}

int SmdMatDB::ReadEoss(CSimpleIni &ini, const int itype) {

	// we already know that the section exists
	string section = gProps[itype].eosName;
	int EosId = ini.GetLongValue(section.c_str(), "EosId", LONG_NOT_FOUND);
	if (EosId == LONG_NOT_FOUND) {
		printf("Missing EosId in eos model\n");
		return -1;
	} else {
		if (EosId == 1) {
			// this is the Linear EOS
			return ReadEosLinear(ini, itype);
		} else {
			printf("EosId = %d unknown\n", EosId);
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

int SmdMatDB::ReadStrengths(CSimpleIni &ini, const int itype) {

	// we already know that the section exists
	string section = gProps[itype].strengthName;
	int MatId = ini.GetLongValue(section.c_str(), "MatId", LONG_NOT_FOUND);
	if (MatId == LONG_NOT_FOUND) {
		printf("Missing MatId in strength model\n");
		return -1;
	} else {
		if (MatId == 1) {
			// this is the Linear EOS
			return ReadStrengthLinear(ini, itype);
		} else {
			printf("MatId = %d unknown\n", MatId);
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

void SmdMatDB::PrintData() {

	for (int itype = 1; itype < ntypes + 1; itype++) {

		printf("\ntype = %d\n", itype);
		printf("material strength name is %s\n", gProps[itype].strengthName.c_str());
		int strengthType = gProps[itype].strengthType;
		int idx = gProps[itype].strengthTypeIdx;

		if (strengthType == 1) { // linear elastic
			printf("number of linear elastic material types: %lu\n", strengthLinear_vec.size());
			printf("linear elastic youngs modulus %f\n", strengthLinear_vec[idx].E);
			printf("linear elastic shear  modulus %f\n", strengthLinear_vec[idx].G);
			printf("linear elastic Poisson ratio  %f\n", strengthLinear_vec[idx].nu);
		}

	}

}

void SmdMatDB::ComputePressure(const double mu, const double temperature, const int itype,
		double &pressure, double &K_eff) {

	int eosType = gProps[itype].eosType;
	int eosIdx = gProps[itype].eosTypeIdx;

	if (eosType == 1) { // linear EOS
		pressure = eosLinear_vec[eosIdx].ComputePressure(mu);
		K_eff = eosLinear_vec[eosIdx].K;
	}
}

void SmdMatDB::ComputeDevStressIncrement(const Matrix3d d_dev, const int itype,
		Matrix3d &stressIncrement) {

	int strengthType = gProps[itype].strengthType;
	int strengthIdx = gProps[itype].strengthTypeIdx;

	if (strengthType == 1) { // linear strength model
		stressIncrement = strengthLinear_vec[strengthIdx].ComputeStressIncrement(d_dev);
	}
}

