/*
 * SmdMatDB.h
 *
 *  Created on: Jan 11, 2016
 *      Author: ganzenmueller
 */

#include "SimpleIni.h"
#include <map>
#include <string>
#include <vector>
#include <Eigen/Eigen>

#ifndef SMDMATDB_H_
#define SMDMATDB_H_

class SmdMatDB {
public:
	SmdMatDB();
	virtual ~SmdMatDB();
	int ReadMaterials(const int ntypes);
	int ReadSectionGeneral(CSimpleIni &ini, const int itype);
	void ComputePressure(const double mu, const double temperature, const int itype,
			double &pressure, double &K_eff);
	void ComputeDevStressIncrement(const Eigen::Matrix3d d_dev, const int itype,
			Eigen::Matrix3d &StressIncrement);
	void PrintData();

	// --------
	int ReadEoss(CSimpleIni &ini, const int itype);
	int ReadEosLinear(CSimpleIni &ini, const int itype);

	// --------
	int ReadStrengths(CSimpleIni &ini, const int itype);
	int ReadStrengthLinear(CSimpleIni &ini, const int itype);

	//CSimpleIni ini;

	// general material properties
	struct gProp {

		gProp() {
			rho0 = 0.0;
			c0 = 0.0;
			cp = 0.0;
			strengthType = 0;
		}

		double rho0;
		double c0;
		double cp; // heat capacity
		int eosType, strengthType;
		int eosTypeIdx, strengthTypeIdx;
		std::string strengthName, eosName;

	};
	gProp *gProps;

	class EosLinear {
		public:
			double K;
			std::string name;
			EosLinear(double K__, std::string name__) {
				K = K__;
				name = name__;
			}
			double ComputePressure(double mu) {
				return 0.0;
			}
		};
	std::vector<EosLinear> eosLinear_vec; // holds all linear eos models in this simulation

	class StrengthLinear {
	public:
		double E, nu, G;
		std::string name;
		StrengthLinear(double E__, double nu__, std::string name__) {
			E = E__;
			nu = nu__;
			G = E__ / (2.*(1. + nu__));
			name = name__;
		}

		Eigen::Matrix3d ComputeStressIncrement(const Eigen::Matrix3d d_dev) {
			return 2.0 * G * d_dev;
		}
	};
	std::vector<StrengthLinear> strengthLinear_vec; // holds all linear strength models in this simulation

private:
	int ntypes;

};

#endif /* SMDMATDB_H_ */
