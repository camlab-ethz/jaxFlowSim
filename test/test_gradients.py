import unittest
import jax
import sys

sys.path.append("/home/diego/studies/uni/thesis_maths/jaxFlowSim")
print("Updated sys.path:", sys.path)
from src.model import runSimulation, runSimulationUnsafe
from src.bifurcations import solveBifurcation
from src.anastomosis import solve_anastomosis
from src.boundary_conditions import setInletBC, setOutletBC
import time
import os
from functools import partial
from jax import block_until_ready, jit
import numpy as np
from jax import grad
from jax.test_util import check_grads
import jax.numpy as jnp

os.chdir(os.path.dirname(__file__) + "/..")

jax.config.update("jax_enable_x64", True)


class TestModels(unittest.TestCase):

    def test_gradients(self):
        # Define a wrapper to test gradients with respect to a subset of parameters
        def solveBifurcation_wrapper(params):
            u1, u2, u3 = params[:3]
            A1, A2, A3 = params[3:6]
            A01, A02, A03 = params[6:9]
            beta1, beta2, beta3 = params[9:12]
            gamma1, gamma2, gamma3 = params[12:15]
            Pext1, Pext2, Pext3 = params[15:18]

            return solveBifurcation(
                u1,
                u2,
                u3,
                A1,
                A2,
                A3,
                A01,
                A02,
                A03,
                beta1,
                beta2,
                beta3,
                gamma1,
                gamma2,
                gamma3,
                Pext1,
                Pext2,
                Pext3,
            )

        # Define parameters
        params = np.array(
            [
                1.0,
                2.0,
                3.0,  # u1, u2, u3
                1.0,
                1.5,
                2.0,  # A1, A2, A3
                0.5,
                0.5,
                0.5,  # A01, A02, A03
                0.1,
                0.1,
                0.1,  # beta1, beta2, beta3
                1.0,
                1.0,
                1.0,  # gamma1, gamma2, gamma3
                0.0,
                0.0,
                0.0,
            ]
        )  # Pext1, Pext2, Pext3

        # Compute gradients
        check_grads(solveBifurcation_wrapper, (params,), order=1)

        def solveAnastomosis_wrapper(params):
            u1, u2, u3 = params[:3]
            A1, A2, A3 = params[3:6]
            A01, A02, A03 = params[6:9]
            beta1, beta2, beta3 = params[9:12]
            gamma1, gamma2, gamma3 = params[12:15]
            Pext1, Pext2, Pext3 = params[15:18]

            return solve_anastomosis(
                u1,
                u2,
                u3,
                A1,
                A2,
                A3,
                A01,
                A02,
                A03,
                beta1,
                beta2,
                beta3,
                gamma1,
                gamma2,
                gamma3,
                Pext1,
                Pext2,
                Pext3,
            )

        # Define parameters
        params = np.array(
            [
                1.0,
                2.0,
                3.0,  # u1, u2, u3
                1.0,
                1.5,
                2.0,  # A1, A2, A3
                0.5,
                0.5,
                0.5,  # A01, A02, A03
                0.1,
                0.1,
                0.1,  # beta1, beta2, beta3
                1.0,
                1.0,
                1.0,  # gamma1, gamma2, gamma3
                0.0,
                0.0,
                0.0,
            ]
        )  # Pext1, Pext2, Pext3

        # Compute gradients and check them
        check_grads(solveAnastomosis_wrapper, (params,), order=1)

        def solveAnastomosis_wrapper(params):
            u1, u2, u3 = params[:3]
            A1, A2, A3 = params[3:6]
            A01, A02, A03 = params[6:9]
            beta1, beta2, beta3 = params[9:12]
            gamma1, gamma2, gamma3 = params[12:15]
            Pext1, Pext2, Pext3 = params[15:18]

            return solve_anastomosis(
                u1,
                u2,
                u3,
                A1,
                A2,
                A3,
                A01,
                A02,
                A03,
                beta1,
                beta2,
                beta3,
                gamma1,
                gamma2,
                gamma3,
                Pext1,
                Pext2,
                Pext3,
            )

        # Define parameters
        params = np.array(
            [
                1.0,
                2.0,
                3.0,  # u1, u2, u3
                1.0,
                1.5,
                2.0,  # A1, A2, A3
                0.5,
                0.5,
                0.5,  # A01, A02, A03
                0.1,
                0.1,
                0.1,  # beta1, beta2, beta3
                1.0,
                1.0,
                1.0,  # gamma1, gamma2, gamma3
                0.0,
                0.0,
                0.0,
            ]
        )  # Pext1, Pext2, Pext3

        # Compute gradients and check them
        check_grads(solveAnastomosis_wrapper, (params,), order=1)

        def setInletBC_wrapper(params):
            inlet = 1
            u0, u1 = params[0:2]
            A = params[2]
            c0, c1 = params[3:5]
            t = params[5]
            dt = params[6]
            cardiac_T = params[7]
            invDx = params[8]
            A0 = params[9]
            beta = params[10]
            P_ext = params[11]
            # Input data in the provided format
            input_data = jnp.array(
                [
                    [0.000000000000000000e00, 1.209639445207363567e-05],
                    [4.899999999999999842e-03, 1.618487866641070779e-05],
                    [9.900000000000000813e-03, 2.107260389974507507e-05],
                    [1.480000000000000066e-02, 2.679179544493246455e-05],
                    [1.980000000000000163e-02, 3.352786925204760611e-05],
                    [2.469999999999999973e-02, 4.090303745212071435e-05],
                    [2.970000000000000070e-02, 4.949101970548220480e-05],
                    [3.459999999999999881e-02, 5.866436284323188656e-05],
                    [3.960000000000000325e-02, 6.910506260529067195e-05],
                    [4.449999999999999789e-02, 8.001195593412976318e-05],
                    [4.939999999999999947e-02, 9.193010834895197134e-05],
                    [5.439999999999999697e-02, 1.046235949172111339e-04],
                    [5.929999999999999855e-02, 1.179710239733674799e-04],
                    [6.429999999999999605e-02, 1.321946524706533815e-04],
                    [6.919999999999999762e-02, 1.463338864274971152e-04],
                    [7.420000000000000207e-02, 1.613916339715019369e-04],
                    [7.910000000000000364e-02, 1.761014769261140664e-04],
                    [8.409999999999999420e-02, 1.914970185909302544e-04],
                    [8.899999999999999578e-02, 2.062706188523366602e-04],
                    [9.389999999999999736e-02, 2.211671176577674318e-04],
                    [9.890000000000000180e-02, 2.358031529730443391e-04],
                    [1.038000000000000034e-01, 2.500123035948267187e-04],
                    [1.087999999999999939e-01, 2.640035907711743438e-04],
                    [1.136999999999999955e-01, 2.768425032606843363e-04],
                    [1.187000000000000000e-01, 2.895065935979986385e-04],
                    [1.236000000000000015e-01, 3.009333354807595734e-04],
                    [1.285999999999999921e-01, 3.120104212098151171e-04],
                    [1.335000000000000075e-01, 3.218214972078619804e-04],
                    [1.385000000000000120e-01, 3.311563221253168029e-04],
                    [1.433999999999999997e-01, 3.392542029329852910e-04],
                    [1.482999999999999874e-01, 3.466409343182314537e-04],
                    [1.532999999999999918e-01, 3.531609712301044110e-04],
                    [1.582000000000000073e-01, 3.588006735128411949e-04],
                    [1.632000000000000117e-01, 3.636868914798052871e-04],
                    [1.680999999999999994e-01, 3.675416639854783450e-04],
                    [1.731000000000000039e-01, 3.707249244862650581e-04],
                    [1.779999999999999916e-01, 3.729779135990776598e-04],
                    [1.829999999999999960e-01, 3.745253420298203829e-04],
                    [1.879000000000000115e-01, 3.752334249817801176e-04],
                    [1.927999999999999992e-01, 3.752028215320961932e-04],
                    [1.978000000000000036e-01, 3.744185846160934424e-04],
                    [2.026999999999999913e-01, 3.728977365573681751e-04],
                    [2.076999999999999957e-01, 3.706110089356431524e-04],
                    [2.126000000000000112e-01, 3.677185301059006958e-04],
                    [2.175999999999999879e-01, 3.640429474492879380e-04],
                    [2.225000000000000033e-01, 3.598761447373066344e-04],
                    [2.275000000000000078e-01, 3.549500407080040604e-04],
                    [2.323999999999999955e-01, 3.496673004412826496e-04],
                    [2.373000000000000109e-01, 3.437989856011837062e-04],
                    [2.422999999999999876e-01, 3.374801550540888458e-04],
                    [2.472000000000000031e-01, 3.307881024603321479e-04],
                    [2.521999999999999797e-01, 3.236196655818732865e-04],
                    [2.570999999999999952e-01, 3.164582404204011284e-04],
                    [2.620999999999999996e-01, 3.087853695300681175e-04],
                    [2.670000000000000151e-01, 3.012213781522257413e-04],
                    [2.720000000000000195e-01, 2.931998036028122330e-04],
                    [2.768999999999999795e-01, 2.853490676815343221e-04],
                    [2.817999999999999949e-01, 2.772200439581621483e-04],
                    [2.867999999999999994e-01, 2.689402201917486879e-04],
                    [2.917000000000000148e-01, 2.605194707121163294e-04],
                    [2.967000000000000193e-01, 2.517332885811343277e-04],
                    [3.015999999999999792e-01, 2.430739972253209106e-04],
                    [3.065999999999999837e-01, 2.338111059835837812e-04],
                    [3.114999999999999991e-01, 2.246075560486266982e-04],
                    [3.165000000000000036e-01, 2.147026809527956075e-04],
                    [3.214000000000000190e-01, 2.048172139488639023e-04],
                    [3.264000000000000234e-01, 1.941580309900602021e-04],
                    [3.312999999999999834e-01, 1.835285942008580275e-04],
                    [3.361999999999999988e-01, 1.723393682810206743e-04],
                    [3.412000000000000033e-01, 1.608032759698219171e-04],
                    [3.461000000000000187e-01, 1.490144201553551622e-04],
                    [3.511000000000000232e-01, 1.367753847095228416e-04],
                    [3.559999999999999831e-01, 1.249153877199872661e-04],
                    [3.609999999999999876e-01, 1.125939311170513790e-04],
                    [3.659000000000000030e-01, 1.008781719374321793e-04],
                    [3.709000000000000075e-01, 8.895994908316577664e-05],
                    [3.758000000000000229e-01, 7.789381693690013424e-05],
                    [3.806999999999999829e-01, 6.713179973840644409e-05],
                    [3.856999999999999873e-01, 5.699570460431591855e-05],
                    [3.906000000000000028e-01, 4.762629000119651977e-05],
                    [3.956000000000000072e-01, 3.891228498797771825e-05],
                    [4.005000000000000226e-01, 3.144183635026534849e-05],
                    [4.055000000000000271e-01, 2.461788947011819365e-05],
                    [4.103999999999999870e-01, 1.901203605594334680e-05],
                    [4.153999999999999915e-01, 1.412838905695585065e-05],
                    [4.203000000000000069e-01, 1.033921179969490092e-05],
                    [4.252000000000000224e-01, 7.308197312096451812e-06],
                    [4.302000000000000268e-01, 5.041943743836069722e-06],
                    [4.350999999999999868e-01, 3.455477536112463137e-06],
                    [4.400999999999999912e-01, 2.419233685391880393e-06],
                    [4.450000000000000067e-01, 1.886855658895387274e-06],
                    [4.500000000000000111e-01, 1.687060570131983003e-06],
                    [4.549000000000000266e-01, 1.747054323492662290e-06],
                    [4.598999999999999755e-01, 1.967150421481014368e-06],
                    [4.647999999999999909e-01, 2.257579748899890353e-06],
                    [4.697000000000000064e-01, 2.566472711299963714e-06],
                    [4.747000000000000108e-01, 2.842748176413116354e-06],
                    [4.796000000000000263e-01, 3.055527885877527876e-06],
                    [4.845999999999999752e-01, 3.195243405772681805e-06],
                    [4.894999999999999907e-01, 3.255947534282193563e-06],
                    [4.944999999999999951e-01, 3.261205322637179941e-06],
                    [4.994000000000000106e-01, 3.233793185450059535e-06],
                    [5.043999999999999595e-01, 3.205660294351655291e-06],
                    [5.092999999999999750e-01, 3.213973306702736651e-06],
                    [5.142999999999999794e-01, 3.294216403553436553e-06],
                    [5.191999999999999948e-01, 3.474459988738417973e-06],
                    [5.241000000000000103e-01, 3.785760763659985098e-06],
                    [5.291000000000000147e-01, 4.246055590588901287e-06],
                    [5.340000000000000302e-01, 4.863696742024284385e-06],
                    [5.390000000000000346e-01, 5.655849707315220050e-06],
                    [5.439000000000000501e-01, 6.565509533360231922e-06],
                    [5.489000000000000545e-01, 7.641757139966823625e-06],
                    [5.537999999999999590e-01, 8.781449213794438989e-06],
                    [5.587999999999999634e-01, 1.003997257213346321e-05],
                    [5.636999999999999789e-01, 1.128919442935887205e-05],
                    [5.685999999999999943e-01, 1.256450096024269509e-05],
                    [5.735999999999999988e-01, 1.380920089788885270e-05],
                    [5.785000000000000142e-01, 1.498589881469865659e-05],
                    [5.835000000000000187e-01, 1.609041683833857899e-05],
                    [5.884000000000000341e-01, 1.703231972209684608e-05],
                    [5.934000000000000385e-01, 1.787365627672909025e-05],
                    [5.983000000000000540e-01, 1.853301899521765086e-05],
                    [6.032999999999999474e-01, 1.906245176496673560e-05],
                    [6.081999999999999629e-01, 1.941517374007269245e-05],
                    [6.130999999999999783e-01, 1.962536547259630132e-05],
                    [6.180999999999999828e-01, 1.969154829020829800e-05],
                    [6.229999999999999982e-01, 1.962410697376247329e-05],
                    [6.280000000000000027e-01, 1.943189436602352923e-05],
                    [6.329000000000000181e-01, 1.913987209780597918e-05],
                    [6.379000000000000226e-01, 1.874238321591428070e-05],
                    [6.428000000000000380e-01, 1.827921556672668245e-05],
                    [6.478000000000000425e-01, 1.772856469310392640e-05],
                    [6.526999999999999469e-01, 1.714081268716011658e-05],
                    [6.575999999999999623e-01, 1.649437119358814326e-05],
                    [6.625999999999999668e-01, 1.580711767924675181e-05],
                    [6.674999999999999822e-01, 1.508953874049739769e-05],
                    [6.724999999999999867e-01, 1.433276656049025933e-05],
                    [6.774000000000000021e-01, 1.359028791862800265e-05],
                    [6.824000000000000066e-01, 1.281125886420697962e-05],
                    [6.873000000000000220e-01, 1.206387595912165379e-05],
                    [6.923000000000000265e-01, 1.129764691417827479e-05],
                    [6.972000000000000419e-01, 1.058094331146793475e-05],
                    [7.022000000000000464e-01, 9.865068958474946671e-06],
                    [7.070999999999999508e-01, 9.214459792930219503e-06],
                    [7.119999999999999662e-01, 8.595912729045065596e-06],
                    [7.169999999999999707e-01, 8.027411127850392941e-06],
                    [7.218999999999999861e-01, 7.514595102018225856e-06],
                    [7.268999999999999906e-01, 7.047906266405176905e-06],
                    [7.318000000000000060e-01, 6.653709718337781729e-06],
                    [7.368000000000000105e-01, 6.293511506837026936e-06],
                    [7.417000000000000259e-01, 5.989705707336698207e-06],
                    [7.467000000000000304e-01, 5.706818795631589092e-06],
                    [7.516000000000000458e-01, 5.456792899948071877e-06],
                    [7.564999999999999503e-01, 5.211506539570724973e-06],
                    [7.614999999999999547e-01, 4.963558646425184622e-06],
                    [7.663999999999999702e-01, 4.702440530645922262e-06],
                    [7.713999999999999746e-01, 4.412127931866063516e-06],
                    [7.762999999999999901e-01, 4.103066880280873450e-06],
                    [7.812999999999999945e-01, 3.747725401856944770e-06],
                    [7.862000000000000099e-01, 3.372269276356612969e-06],
                    [7.912000000000000144e-01, 2.952014389655351788e-06],
                    [7.961000000000000298e-01, 2.525876764732685429e-06],
                    [8.010000000000000453e-01, 2.081192824111645603e-06],
                    [8.060000000000000497e-01, 1.638976119775932770e-06],
                    [8.108999999999999542e-01, 1.216627379035810466e-06],
                    [8.158999999999999586e-01, 8.219497076753892585e-07],
                    [8.207999999999999741e-01, 4.950546043938223583e-07],
                    [8.257999999999999785e-01, 2.226838612267707405e-07],
                    [8.306999999999999940e-01, 4.013098373176595924e-08],
                    [8.356999999999999984e-01, -6.145240644951676240e-08],
                    [8.406000000000000139e-01, -6.713024886386509919e-08],
                    [8.455000000000000293e-01, 1.654825591877841358e-08],
                    [8.505000000000000338e-01, 1.857108638077427620e-07],
                    [8.554000000000000492e-01, 4.289557280374676148e-07],
                    [8.604000000000000536e-01, 7.374230256163179110e-07],
                    [8.652999999999999581e-01, 1.073329020585869166e-06],
                    [8.702999999999999625e-01, 1.439820000961674500e-06],
                    [8.751999999999999780e-01, 1.786297263826130370e-06],
                    [8.801999999999999824e-01, 2.116628586932765308e-06],
                    [8.850999999999999979e-01, 2.382542114323806446e-06],
                    [8.901000000000000023e-01, 2.587354503580473605e-06],
                    [8.950000000000000178e-01, 2.697694193560188232e-06],
                    [8.999000000000000332e-01, 2.715047089826767133e-06],
                    [9.049000000000000377e-01, 2.631726685400349554e-06],
                    [9.098000000000000531e-01, 2.451081444566862982e-06],
                    [9.147999999999999465e-01, 2.178177786043121020e-06],
                    [9.196999999999999620e-01, 1.848088895124506421e-06],
                    [9.246999999999999664e-01, 1.463951752031612731e-06],
                    [9.295999999999999819e-01, 1.087626502039134374e-06],
                    [9.345999999999999863e-01, 7.317663458237682579e-07],
                    [9.395000000000000018e-01, 4.726511382674197754e-07],
                    [9.444000000000000172e-01, 3.445214160388589785e-07],
                    [9.494000000000000217e-01, 4.136968478020873701e-07],
                    [9.543000000000000371e-01, 7.443747515661756851e-07],
                    [9.593000000000000416e-01, 1.418020346082462601e-06],
                    [9.641999999999999460e-01, 2.463993722833809842e-06],
                    [9.691999999999999504e-01, 4.023307412154667662e-06],
                    [9.740999999999999659e-01, 6.062508785098040651e-06],
                    [9.790999999999999703e-01, 8.794551648845374016e-06],
                    [9.839999999999999858e-01, 1.209639445207364414e-05],
                ]
            )

            return setInletBC(u0, u1, A, c0, c1, t, dt, input_data, cardiac_T, invDx)

        # Define parameters
        params = jnp.array(
            [  # 1.0,             # inlet
                2.0,
                3.0,  # u0, u1
                1.5,  # A
                0.1,
                0.2,  # c0, c1
                0.5,  # t
                0.01,  # dt
                1.0,  # cardiac_T
                0.05,  # invDx
                0.5,  # A0
                0.2,  # beta
                0.0,
            ]
        )

        # Compute gradients and check them
        check_grads(setInletBC_wrapper, (params,), order=1)


if __name__ == "__main__":
    unittest.main()
