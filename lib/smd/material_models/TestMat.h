/*
 * TestMat.h
 *
 *  Created on: Feb 4, 2016
 *      Author: ganzenmueller
 */

#ifndef TESTMAT_H_
#define TESTMAT_H_

class StrengthGCG {
public:
	double E, nu, G, yieldStress;
	std::string name;
	StrengthGCG(double E__, double nu__, double yieldStress__, std::string name__) {
		E = E__;
		nu = nu__;
		G = E__ / (2. * (1. + nu__));
		yieldStress = yieldStress__;
		name = name__;
	}

	void ComputeStressIncrement(const Eigen::Matrix3d deviatoricStrainIncrement, const Eigen::Matrix3d oldStressDeviator,
			double &plastic_strain_increment, Eigen::Matrix3d &deviatoricStressIncrement, double &plastic_work) {

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
			plastic_strain_increment = plastic_work = 0.0;
			//printf("no yield\n");

		} else {

			/*
			 * yielding has occured
			 */
			//plastic_strain_increment = sqrt(3. / 2.) * (J2 - yieldStress) / (3.0 * G);
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

			/*
			 * back-calculate elastic strain from stress increment using linear
			 * behavior. Decompose total strain increment in elastic and plastic parts.
			 */

			//Eigen::Matrix3d elasticStrainIncrement = deviatoricStressIncrement / (2.0 * G);
			//Eigen::Matrix3d plasticStrainIncrement = deviatoricStrainIncrement - elasticStrainIncrement;
			//double ratio = plastic_strain_increment / plasticStrainIncrement.norm();
			//printf("ratio p1/p2 = %f\n", ratio);
			/*
			 * plastic heating: plastic strain increment x yield stress
			 */
			//double heat_increment = plasticStrainIncrement.norm() * yieldStress;
			//double heat_increment = plastic_strain_increment * yieldStress;
			//printf("heat increment (sans volume term) is %f\n", heat_increment);

			sigmaTrial_dev = oldStressDeviator + deviatoricStressIncrement;
			//printf("CHECK, J2=%f should be smaller than yield stress\n", sqrt(3. / 2.) * sigmaTrial_dev.norm());

			plastic_work = plastic_strain_increment * yieldStress;
		}

	}

	void PrintParameters() {
		printf("... This is strength model #2, Simple Plasticity\n");
		printf("... Young's modulus %f\n", E);
		printf("... shear modulus %f\n", G);
		printf("... Poisson ratio %f\n", nu);
		printf("... yield stress %f\n", yieldStress);
	}
};

#endif /* TESTMAT_H_ */
