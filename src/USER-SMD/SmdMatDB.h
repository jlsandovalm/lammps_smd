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
	void ComputePressure(const double mu, const double temperature, const int itype, double &pressure, double &K_eff);
	void ComputeDevStressIncrement(const Eigen::Matrix3d d_dev, const int itype, const Eigen::Matrix3d oldStressDeviator,
			double &plasticStrainIncrement, Eigen::Matrix3d &stressIncrement);
	void ComputeViscousStress(const Eigen::Matrix3d d_dev, const int itype, Eigen::Matrix3d &viscousStress);
	void PrintData();

	// --------
	int ReadEoss(CSimpleIni &ini, const int itype);
	int ReadEosLinear(CSimpleIni &ini, const int itype);

	// --------
	int ReadStrengths(CSimpleIni &ini, const int itype);
	int ReadStrengthLinear(CSimpleIni &ini, const int itype);
	int ReadStrengthSimplePlasticity(CSimpleIni &ini, const int itype);

	// --------
	int ReadViscosities(CSimpleIni &ini, const int itype);
	int ReadViscNewton(CSimpleIni &ini, const int itype);

	//CSimpleIni ini;

	// general material properties
	struct gProp {

		gProp() {
			rho0 = 0.0;
			c0 = 0.0;
			cp = 0.0;
			eosType = strengthType = viscType = 0;
			G0 = 0.0;
			K0 = 0.0;
			nu0 = 0.0;
			strengthName = "NONE";
			eosName = "NONE";
			viscName = "NONE";
		}

		double rho0;
		double c0;
		double cp; // heat capacity
		double K0;
		double G0;
		double nu0;
		int eosType, strengthType, viscType;
		int eosTypeIdx, strengthTypeIdx, viscTypeIdx;
		std::string strengthName, eosName, viscName;

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
			return K * mu;
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
			G = E__ / (2. * (1. + nu__));
			name = name__;
		}

		Eigen::Matrix3d ComputeStressIncrement(const Eigen::Matrix3d deviatoricStrainIncrement) {
			return 2.0 * G * deviatoricStrainIncrement;
		}
	};
	std::vector<StrengthLinear> strengthLinear_vec; // holds all linear strength models in this simulation

	class StrengthSimplePlasticity {
	public:
		double E, nu, G, yieldStress;
		std::string name;
		StrengthSimplePlasticity(double E__, double nu__, double yieldStress__, std::string name__) {
			E = E__;
			nu = nu__;
			G = E__ / (2. * (1. + nu__));
			yieldStress = yieldStress__;
			name = name__;
		}

		void ComputeStressIncrement(const Eigen::Matrix3d deviatoricStrainIncrement, const Eigen::Matrix3d oldStressDeviator,
				double &plastic_strain_increment, Eigen::Matrix3d &deviatoricStressIncrement) {

			/*
			 * perform a trial elastic update to the deviatoric stress
			 */
			deviatoricStressIncrement = 2.0 * G * deviatoricStrainIncrement;
			Eigen::Matrix3d sigmaTrial_dev = oldStressDeviator + deviatoricStressIncrement;

			/*
			 * check yield condition
			 */
			double J2 = sqrt(3. / 2.) * sigmaTrial_dev.norm();

			if (J2 < yieldStress) {
				/*
				 * no yielding has occured.
				 * final deviatoric stress is trial deviatoric stress
				 */
				plastic_strain_increment = 0.0;
				//printf("no yield\n");

			} else {

				/*
				 * yielding has occured
				 */
				plastic_strain_increment = (J2 - yieldStress) / (3.0 * G);
				//printf("yield, plastic strain increment is %f, J2=%f, yield stress=%f\n", plastic_strain_increment, J2, yieldStress);
				/*
				 * new deviatoric stress:
				 * obtain by scaling the trial stress deviator
				 */
				//Eigen::Matrix3d sigmaScaled = (yieldStress / J2) * sigmaTrial_dev;
				sigmaTrial_dev *= (yieldStress / J2);

				/*
				 * new deviatoric stress increment
				 */
				deviatoricStressIncrement = sigmaTrial_dev - oldStressDeviator;

				sigmaTrial_dev = oldStressDeviator + deviatoricStressIncrement;
				//printf("CHECK, J2=%f should be smaller than yield stress\n", sqrt(3. / 2.) * sigmaTrial_dev.norm());
			}

		}
	};
	std::vector<StrengthSimplePlasticity> strengthSimplePlasticity_vec; // holds all linear strength models in this simulation

	class ViscosityNewton {
	public:
		double eta;
		std::string name;
		ViscosityNewton(double eta__, std::string name__) {
			eta = eta__;
			name = name__;
		}

		Eigen::Matrix3d ComputeStressDeviator(const Eigen::Matrix3d strainRateDeviator) {
			return 2.0 * eta * strainRateDeviator;
		}
	};
	std::vector<ViscosityNewton> viscNewton_vec; // holds all Newton viscosity models in this simulation

private:
	int ntypes;

};

#endif /* SMDMATDB_H_ */
