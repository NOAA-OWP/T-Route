# Commenting out due to fortan pointer dealloc errors
# ---------------------------------------------------
#  import os
# from array import array
# from pathlib import Path

# import pytest

# from troute.network.reservoirs.levelpool.levelpool import MC_Levelpool
# from troute.network.reservoirs.rfc.rfc import MC_RFC 


# @pytest.fixture
# def lp_reservoir():
#     """
#     an lp compute kernel to put under test
#     """
#     #print("lp test")

#     water_elevation = 9.7373
#     lake_area = 15.0949
#     weir_elevation = 9.626
#     weir_coefficient = 0.4
#     weir_length = 10.0
#     dam_length = 10.0
#     orifice_elevation = 7.733
#     orifice_coefficient = 0.1
#     orifice_area = 1.0
#     max_depth = 9.96
#     lake_number = 16944276
#     initial_fractional_depth = 0.9
#     args = [lake_area, max_depth, orifice_area,
#             orifice_coefficient, orifice_elevation,
#             weir_coefficient, weir_elevation, weir_length,
#             initial_fractional_depth, 0.0, water_elevation]

#     upstream_ids = array('l')
#     k = MC_Levelpool(0, lake_number, upstream_ids, args)
#     return k


# @pytest.fixture
# def lp_reservoir2():
#     """
#     an lp compute kernel to put under test
#     """
#     #print("lp2 test")

#     water_elevation = 9.70
#     lake_area = 15.0949
#     weir_elevation = 9.626
#     weir_coefficient = 0.4
#     weir_length = 10.0
#     dam_length = 10.0
#     orifice_elevation = 7.733
#     orifice_coefficient = 0.1
#     orifice_area = 1.0
#     max_depth = 9.96
#     lake_number = 16944277
#     initial_fractional_depth = 0.9
#     args = [lake_area, max_depth, orifice_area,
#             orifice_coefficient, orifice_elevation,
#             weir_coefficient, weir_elevation, weir_length,
#             initial_fractional_depth, 0.0, water_elevation]

#     upstream_ids = array('l')
#     k = MC_Levelpool(0, lake_number, upstream_ids, args)
#     return k


# @pytest.fixture
# def lp_reservoir3():
#     """
#     Lake number 4185105 from Pocono test dataset
#     an lp compute kernel to put under test
#     """
#     #print("lp3 test")

#     water_elevation = 496.25399574057275
#     lake_area = 1.95502996444702
#     weir_elevation = 496.19599609375
#     weir_coefficient = 0.4
#     weir_length = 10.0
#     dam_length = 10.0
#     orifice_elevation = 495.210001627604
#     orifice_coefficient = 0.1
#     orifice_area = 1.0
#     max_depth = 496.369995117188
#     lake_number = 4185105
#     initial_fractional_depth = 0.9
#     args = [lake_area, max_depth, orifice_area,
#             orifice_coefficient, orifice_elevation,
#             weir_coefficient, weir_elevation, weir_length,
#             initial_fractional_depth, 0.0, water_elevation]

#     upstream_ids = array('l')
#     k = MC_Levelpool(0, lake_number, upstream_ids, args)
#     return k

# @pytest.fixture
# def rfc_reservoir():
#     """
#     an lp compute kernel to put under test
#     """
#     #print("rfc test")

#     cwd_full = Path(__file__).parent / "data/reservoir_testing_files/"

#     water_elevation = 1331.18005
#     lake_area = 209.632
#     weir_elevation = 1332.074
#     weir_coefficient = 0.4
#     weir_length = 10.0
#     dam_length = 10.0
#     orifice_elevation = 1314.473
#     orifice_coefficient = 0.1
#     orifice_area = 1.0
#     max_depth = 1335.180
#     initial_fractional_depth = 0.9
#     lake_number = 17609317
#     reservoir_type = 4
#     reservoir_parameter_file = os.path.join(cwd_full, "reservoir_index_short_range.nc")
#     start_date = "2019-08-18_09:00:00"
#     time_series_path = cwd_full
#     forecast_lookback_hours = 24

#     args = [lake_area, max_depth, orifice_area,
#             orifice_coefficient, orifice_elevation,
#             weir_coefficient, weir_elevation, weir_length,
#             initial_fractional_depth, 0.0, water_elevation]

#     upstream_ids = array('l')
#     k = MC_RFC(0, lake_number, upstream_ids, args,
#                reservoir_type, reservoir_parameter_file,
#                start_date, time_series_path, forecast_lookback_hours)
#     return k


# def test_lp_construction(lp_reservoir):
#     """
#     test construction of an lp reservoir 
#     """
#     water_elevation = 9.7373
#     lake_area = 15.0949
#     weir_elevation = 9.626
#     weir_coefficient = 0.4
#     weir_length = 10.0
#     dam_length = 10.0
#     orifice_elevation = 7.733
#     orifice_coefficient = 0.1
#     orifice_area = 1.0
#     max_depth = 9.96
#     lake_number = 16944276
#     initial_fractional_depth = 0.9

#     assert(lp_reservoir.water_elevation == pytest.approx(water_elevation, rel=1e-7) )
#     assert(lp_reservoir.lake_area == pytest.approx(lake_area, rel=1e-7) )
#     assert(lp_reservoir.weir_elevation == pytest.approx(weir_elevation, rel=1e-7) )
#     assert(lp_reservoir.weir_coefficient == pytest.approx(weir_coefficient, rel=1e-7) )
#     assert(lp_reservoir.weir_length == pytest.approx(weir_length, rel=1e-7) )
#     assert(lp_reservoir.dam_length == pytest.approx(dam_length, rel=1e-7) )
#     assert(lp_reservoir.orifice_elevation == pytest.approx(orifice_elevation, rel=1e-7) )
#     assert(lp_reservoir.orifice_area == pytest.approx(orifice_area, rel=1e-7) )
#     assert(lp_reservoir.max_depth == pytest.approx(max_depth, rel=1e-7) )
#     assert(lp_reservoir.lake_number == lake_number )
#     assert(lp_reservoir.initial_fractional_depth == pytest.approx(initial_fractional_depth, rel=1e-7) )


# def test_lp_construction2(lp_reservoir2):
#     """
#     test construction of an lp reservoir 
#     """
#     water_elevation = 9.70
#     lake_area = 15.0949
#     weir_elevation = 9.626
#     weir_coefficient = 0.4
#     weir_length = 10.0
#     dam_length = 10.0
#     orifice_elevation = 7.733
#     orifice_coefficient = 0.1
#     orifice_area = 1.0
#     max_depth = 9.96
#     lake_number = 16944277
#     initial_fractional_depth = 0.9

#     assert(lp_reservoir2.water_elevation == pytest.approx(water_elevation, rel=1e-7) )

#     assert(lp_reservoir2.lake_area == pytest.approx(lake_area, rel=1e-7) )
#     assert(lp_reservoir2.weir_elevation == pytest.approx(weir_elevation, rel=1e-7) )
#     assert(lp_reservoir2.weir_coefficient == pytest.approx(weir_coefficient, rel=1e-7) )
#     assert(lp_reservoir2.weir_length == pytest.approx(weir_length, rel=1e-7) )
#     assert(lp_reservoir2.dam_length == pytest.approx(dam_length, rel=1e-7) )
#     assert(lp_reservoir2.orifice_elevation == pytest.approx(orifice_elevation, rel=1e-7) )
#     assert(lp_reservoir2.orifice_area == pytest.approx(orifice_area, rel=1e-7) )
#     assert(lp_reservoir2.max_depth == pytest.approx(max_depth, rel=1e-7) )
#     assert(lp_reservoir2.lake_number == lake_number )
#     assert(lp_reservoir2.initial_fractional_depth == pytest.approx(initial_fractional_depth, rel=1e-7) )


# def test_lp_construction3(lp_reservoir3):
#     """
#     test construction of an lp reservoir 
#     """
#     water_elevation = 496.25399574057275
#     lake_area = 1.95502996444702
#     weir_elevation = 496.19599609375
#     weir_coefficient = 0.4
#     weir_length = 10.0
#     dam_length = 10.0
#     orifice_elevation = 495.210001627604
#     orifice_coefficient = 0.1
#     orifice_area = 1.0
#     max_depth = 496.369995117188
#     lake_number = 4185105
#     initial_fractional_depth = 0.9

#     assert(lp_reservoir3.water_elevation == pytest.approx(water_elevation, rel=1e-7) )

#     assert(lp_reservoir3.lake_area == pytest.approx(lake_area, rel=1e-7) )
#     assert(lp_reservoir3.weir_elevation == pytest.approx(weir_elevation, rel=1e-7) )
#     assert(lp_reservoir3.weir_coefficient == pytest.approx(weir_coefficient, rel=1e-7) )
#     assert(lp_reservoir3.weir_length == pytest.approx(weir_length, rel=1e-7) )
#     assert(lp_reservoir3.dam_length == pytest.approx(dam_length, rel=1e-7) )
#     assert(lp_reservoir3.orifice_elevation == pytest.approx(orifice_elevation, rel=1e-7) )
#     assert(lp_reservoir3.orifice_area == pytest.approx(orifice_area, rel=1e-7) )
#     assert(lp_reservoir3.max_depth == pytest.approx(max_depth, rel=1e-7) )
#     assert(lp_reservoir3.lake_number == lake_number )
#     assert(lp_reservoir3.initial_fractional_depth == pytest.approx(initial_fractional_depth, rel=1e-7) )



# def test_hybrid_construction(hybrid_reservoir):
#     """
#     test construction of a hybrid reservoir 
#     """

#     cwd = os.path.dirname(os.path.realpath(__file__)).encode('UTF-8')
#     cwd_full = os.path.join(cwd, b"reservoir_testing_files/")

#     water_elevation = 1331.18005
#     lake_area = 209.632
#     weir_elevation = 1332.074
#     weir_coefficient = 0.4
#     weir_length = 10.0
#     dam_length = 10.0
#     orifice_elevation = 1314.473
#     orifice_coefficient = 0.1
#     orifice_area = 1.0
#     max_depth = 1335.180
#     initial_fractional_depth = 0.9
#     lake_number = 402142
#     reservoir_type = 2
#     reservoir_parameter_file = os.path.join(cwd_full, b"reservoir_index_short_range.nc")
#     start_date = b"2010-10-01_07:00:00"
#     usgs_timeslice_path = cwd_full
#     usace_timeslice_path = cwd_full
#     observation_lookback_hours = 48
#     observation_update_time_interval_seconds = 1000000000


#     assert(hybrid_reservoir.water_elevation == pytest.approx(water_elevation, rel=1e-7) )

#     assert(hybrid_reservoir.lake_area == pytest.approx(lake_area, rel=1e-7) )
#     assert(hybrid_reservoir.weir_elevation == pytest.approx(weir_elevation, rel=1e-7) )
#     assert(hybrid_reservoir.weir_coefficient == pytest.approx(weir_coefficient, rel=1e-7) )
#     assert(hybrid_reservoir.weir_length == pytest.approx(weir_length, rel=1e-7) )
#     assert(hybrid_reservoir.dam_length == pytest.approx(dam_length, rel=1e-7) )
#     assert(hybrid_reservoir.orifice_elevation == pytest.approx(orifice_elevation, rel=1e-7) )
#     assert(hybrid_reservoir.orifice_area == pytest.approx(orifice_area, rel=1e-7) )
#     assert(hybrid_reservoir.max_depth == pytest.approx(max_depth, rel=1e-7) )
#     assert(hybrid_reservoir.lake_number == lake_number )
#     assert(hybrid_reservoir.initial_fractional_depth == pytest.approx(initial_fractional_depth, rel=1e-7) )
#     assert(hybrid_reservoir.reservoir_type == reservoir_type )
#     assert(hybrid_reservoir.reservoir_parameter_file == reservoir_parameter_file )
#     assert(hybrid_reservoir.start_date == start_date )
#     assert(hybrid_reservoir.usgs_timeslice_path == usgs_timeslice_path )
#     assert(hybrid_reservoir.usace_timeslice_path == usace_timeslice_path )
#     assert(hybrid_reservoir.observation_lookback_hours == observation_lookback_hours )
#     assert(hybrid_reservoir.observation_update_time_interval_seconds == observation_update_time_interval_seconds )



# def test_rfc_construction(rfc_reservoir):
#     """
#     test construction of a rfc reservoir 
#     """
#     cwd_full = Path(__file__).parent / "data/reservoir_testing_files/"
#     print(cwd_full)

#     water_elevation = 1331.18005
#     lake_area = 209.632
#     weir_elevation = 1332.074
#     weir_coefficient = 0.4
#     weir_length = 10.0
#     dam_length = 10.0
#     orifice_elevation = 1314.473
#     orifice_coefficient = 0.1
#     orifice_area = 1.0
#     max_depth = 1335.180
#     initial_fractional_depth = 0.9
#     lake_number = 17609317
#     reservoir_type = 4
#     reservoir_parameter_file = os.path.join(cwd_full, b"reservoir_index_short_range.nc")
#     start_date = b"2019-08-18_09:00:00"
#     time_series_path = cwd_full
#     forecast_lookback_hours = 24

#     assert(rfc_reservoir.water_elevation == pytest.approx(water_elevation, rel=1e-7) )

#     assert(rfc_reservoir.lake_area == pytest.approx(lake_area, rel=1e-7) )
#     assert(rfc_reservoir.weir_elevation == pytest.approx(weir_elevation, rel=1e-7) )
#     assert(rfc_reservoir.weir_coefficient == pytest.approx(weir_coefficient, rel=1e-7) )
#     assert(rfc_reservoir.weir_length == pytest.approx(weir_length, rel=1e-7) )
#     assert(rfc_reservoir.dam_length == pytest.approx(dam_length, rel=1e-7) )
#     assert(rfc_reservoir.orifice_elevation == pytest.approx(orifice_elevation, rel=1e-7) )
#     assert(rfc_reservoir.orifice_area == pytest.approx(orifice_area, rel=1e-7) )
#     assert(rfc_reservoir.max_depth == pytest.approx(max_depth, rel=1e-7) )
#     assert(rfc_reservoir.lake_number == lake_number )
#     assert(rfc_reservoir.initial_fractional_depth == pytest.approx(initial_fractional_depth, rel=1e-7) )
#     assert(rfc_reservoir.reservoir_type == reservoir_type )
#     assert(rfc_reservoir.reservoir_parameter_file == reservoir_parameter_file )
#     assert(rfc_reservoir.start_date == start_date )
#     assert(rfc_reservoir.time_series_path == time_series_path )
#     assert(rfc_reservoir.forecast_lookback_hours == forecast_lookback_hours )



# def test_lp_run(lp_reservoir):
#     """
#     test running a LP reservoir
#     """

#     inflow_list = [
#         91.27196,
#         91.7394,
#         92.15904,
#         92.1518,
#         91.84663,
#         91.38554,
#         90.86131,
#         90.32736,
#         89.81273,
#         89.3325,
#         88.89427,
#         88.5025,
#         88.16228,
#         87.41539,
#         86.80043,
#         86.03979,
#         85.3849,
#         85.33451,
#         86.84274,
#         91.6084,
#         101.81398,
#         118.85916,
#         143.99232,
#         177.7355,
#         219.2348,
#         267.22351,
#         319.90402,
#         374.54324,
#         428.86066,
#         480.92096,
#         529.23584,
#         572.77673,
#         610.93237,
#         643.4389,
#         670.28516,
#         691.67767,
#         707.96088,
#         719.57312,
#         726.96997,
#         730.63269,
#         731.03186,
#         728.61438,
#         723.79578,
#         716.9549,
#         708.43268,
#         698.53247,
#         687.52112,
#         675.63123,
#         663.06421,
#         649.99976,
#         636.57898,
#         622.92926,
#         609.1745,
#         595.40369,
#         581.68799,
#         568.08588,
#         554.64484,
#         541.4032,
#         528.39185,
#         515.63513,
#         503.14838,
#         490.95123,
#         479.05109,
#         467.45493,
#         456.16663,
#         445.18753,
#         434.51706,
#         424.15311,
#         414.0921,
#         404.32956,
#         394.86014,
#         385.67789,
#         376.77621,
#         368.14966,
#         359.78958,
#         351.68875,
#         343.83972,
#         336.23505,
#         328.86719,
#         321.7287,
#         314.81219,
#         308.11047,
#         301.61646,
#         295.32312,
#         289.22369,
#         283.31207,
#         277.5813,
#         272.02521,
#         266.63776,
#         261.41315,
#         256.34564,
#         251.42978,
#         246.66023,
#         242.03192,
#         237.53989,
#         233.17944,
#         228.94595,
#         224.83511,
#         220.84265,
#         216.96449,
#         213.19672,
#         209.53554,
#         205.97734,
#         202.51857,
#         199.1559,
#         195.88605,
#         192.70595,
#         189.61255,
#     ]

#     routing_period = 300.0

#     for inflow in inflow_list:
#         out, water_elevation = lp_reservoir.run(inflow, 0.0, routing_period)

#         #print(out)
#         #print(water_elevation)

#     expected_final_outflow = 17.0437641
#     expected_final_water_elevation = 10.4923334

#     assert lp_reservoir is not None
#     assert expected_final_outflow == pytest.approx(out)
#     assert expected_final_water_elevation == pytest.approx(water_elevation)



# def test_lp2_run(lp_reservoir2):
#     """
#     test running a LP reservoir
#     """

#     inflow_list = [
#         91.27196,
#         91.7394,
#         92.15904,
#         92.1518,
#         91.84663,
#         91.38554,
#         90.86131,
#         90.32736,
#         89.81273,
#         89.3325,
#         88.89427,
#         88.5025,
#         88.16228,
#         87.41539,
#         86.80043,
#         86.03979,
#         85.3849,
#         85.33451,
#         86.84274,
#         91.6084,
#         101.81398,
#         118.85916,
#         143.99232,
#         177.7355,
#         219.2348,
#         267.22351,
#         319.90402,
#         374.54324,
#         428.86066,
#         480.92096,
#         529.23584,
#         572.77673,
#         610.93237,
#         643.4389,
#         670.28516,
#         691.67767,
#         707.96088,
#         719.57312,
#         726.96997,
#         730.63269,
#         731.03186,
#         728.61438,
#         723.79578,
#         716.9549,
#         708.43268,
#         698.53247,
#         687.52112,
#         675.63123,
#         663.06421,
#         649.99976,
#         636.57898,
#         622.92926,
#         609.1745,
#         595.40369,
#         581.68799,
#         568.08588,
#         554.64484,
#         541.4032,
#         528.39185,
#         515.63513,
#         503.14838,
#         490.95123,
#         479.05109,
#         467.45493,
#         456.16663,
#         445.18753,
#         434.51706,
#         424.15311,
#         414.0921,
#         404.32956,
#         394.86014,
#         385.67789,
#         376.77621,
#         368.14966,
#         359.78958,
#         351.68875,
#         343.83972,
#         336.23505,
#         328.86719,
#         321.7287,
#         314.81219,
#         308.11047,
#         301.61646,
#         295.32312,
#         289.22369,
#         283.31207,
#         277.5813,
#         272.02521,
#         266.63776,
#         261.41315,
#         256.34564,
#         251.42978,
#         246.66023,
#         242.03192,
#         237.53989,
#         233.17944,
#         228.94595,
#         224.83511,
#         220.84265,
#         216.96449,
#         213.19672,
#         209.53554,
#         205.97734,
#         202.51857,
#         199.1559,
#         195.88605,
#         192.70595,
#         189.61255,
#     ]

#     routing_period = 300.0

#     for inflow in inflow_list:
#         out, water_elevation = lp_reservoir2.run(inflow, 0.0, routing_period)

#         #print(out)
#         #print(water_elevation)

#     expected_final_outflow = 15.5038433
#     expected_final_water_elevation = 10.4566612

#     assert lp_reservoir2 is not None
#     assert expected_final_outflow == pytest.approx(out)
#     assert expected_final_water_elevation == pytest.approx(water_elevation)



# def test_lp3_run(lp_reservoir3):
#     """
#     test running a LP reservoir with lake number 4185105 from Pocono test dataset
#     with the test upstream flows from the routing network
#     """

#     inflow_list = [
#         1.80895018577575,
#         1.8080631494522,
#         1.80805230140686,
#         1.80804181098937,
#         1.80803155899047,
#         1.80802226066589,
#         1.80779933929443,
#         1.80708813667297,
#         1.80565989017486,
#         1.80337071418762,
#         1.80015897750854,
#         1.79602336883544,
#         1.7910189628601,
#         1.7852246761322,
#         1.77873408794403,
#         1.77164959907531,
#         1.76404619216918,
#         1.7560269832611,
#         1.74768710136413,
#         1.73911035060882,
#         1.73037016391754,
#         1.72153043746948,
#         1.71264457702636,
#         1.70375370979309,
#         1.69490242004394,
#         1.68612349033355,
#         1.67744469642639,
#         1.66888844966888,
#         1.66047358512878,
#         1.65221524238586,
#         1.64412498474121,
#         1.63621282577514,
#         1.6284852027893,
#         1.6209477186203,
#         1.61360335350036,
#         1.6064486503601,
#         1.59949028491973,
#         1.59272789955139,
#         1.58616030216217,
#         1.57978844642639,
#         1.57360935211181,
#         1.56761956214904,
#         1.56181550025939,
#         1.55619323253631,
#         1.55074894428253,
#         1.54547810554504,
#         1.54037690162658,
#         1.53543543815612,
#         1.53065514564514,
#         1.52603113651275,
#         1.52155971527099,
#         1.51723611354827,
#         1.51230072975158,
#         1.50761890411376,
#         1.50261342525482,
#         1.49800276756286,
#         1.49323236942291,
#         1.4888972043991,
#         1.48448252677917,
#         1.48048961162567,
#         1.48074543476104,
#         1.47737908363342,
#         1.47399830818176,
#         1.47096228599548,
#         1.46792876720428,
#         1.46520555019378,
#         1.46248853206634,
#         1.46004831790924,
#         1.45761549472808,
#         1.45542776584625,
#         1.45324802398681,
#         1.45128095149993,
#         1.44931924343109,
#         1.44755291938781,
#         1.44579112529754,
#         1.44420051574707,
#         1.44261646270751,
#         1.44118237495422,
#         1.43972718715667,
#         1.43840849399566,
#         1.43709254264831,
#         1.43589913845062,
#         1.43470692634582,
#         1.43362259864807,
#         1.43253803253173,
#         1.43155455589294,
#         1.43057394027709,
#         1.42969226837158,
#         1.42880952358245,
#         1.42801165580749,
#         1.42721033096313,
#         1.42648684978485,
#         1.42030370235443,
#         1.4197164773941,
#         1.41912138462066,
#         1.41856682300567,
#         1.4180212020874,
#         1.41752004623413,
#         1.41702127456665,
#         1.41657400131225,
#         1.41611003875732,
#         1.41569292545318,
#         1.41527259349822,
#         1.41489124298095,
#         1.41450810432434,
#         1.41416013240814,
#         1.41380488872528,
#         1.41348576545715,
#         1.4131577014923,
#         1.41286087036132,
#         1.41256809234619,
#         1.41229903697967,
#         1.4120215177536,
#         1.41177535057067,
#         1.41152358055114,
#         1.41128826141357,
#         1.41106629371643,
#         1.41086113452911,
#         1.41065275669097,
#         1.41046237945556,
#         1.41026413440704,
#         1.41008532047271,
#         1.40990281105041,
#         1.40971994400024,
#         1.40954422950744,
#         1.40936529636383,
#         1.40919578075408,
#         1.40903830528259,
#         1.40887701511383,
#         1.4087312221527,
#         1.40859770774841,
#         1.40845239162445,
#         1.40831840038299,
#         1.40819752216339,
#         1.40807390213012,
#         1.40795993804931,
#         1.40786027908325,
#         1.4077603816986,
#         1.40766382217407,
#         1.40757846832275,
#         1.4074900150299,
#         1.40741646289825,
#         1.40733742713928,
#         1.40725946426391,
#         1.40718698501586,
#         1.40712130069732,
#         1.40705502033233,
#         1.40700340270996,
#         1.40694653987884,
#         1.40689206123352,
#         1.40683484077453,
#         1.40678656101226,
#         1.40673542022705,
#         1.40668964385986,
#         1.40665125846862,
#         1.40661370754241,
#         1.40657091140747,
#         1.40652787685394,
#         1.40649056434631,
#         1.40644967555999,
#         1.40641677379608,
#         1.40638422966003,
#         1.4063503742218,
#         1.40632283687591,
#         1.40629792213439,
#         1.40626764297485,
#         1.40624809265136,
#         1.40621292591094,
#         1.40618431568145,
#         1.40615904331207,
#         1.40612769126892,
#         1.40610027313232,
#         1.40607714653015,
#         1.4060536623001,
#         1.40603530406951,
#         1.40601682662963,
#         1.40599799156188,
#         1.40597963333129,
#         1.40596461296081,
#         1.40593695640563,
#         1.40591478347778,
#         1.4058940410614,
#         1.40587365627288,
#         1.40586030483245,
#         1.40583717823028,
#         1.40582168102264,
#         1.40580356121063,
#         1.40578985214233,
#         1.40577566623687,
#         1.40575957298278,
#         1.40574049949645,
#         1.40572667121887,
#         1.40571069717407,
#         1.40569508075714,
#         1.40567672252655,
#         1.40565741062164,
#         1.40564179420471,
#         1.40563249588012,
#         1.4056133031845,
#         1.40559816360473,
#         1.40558767318725,
#         1.40557587146759,
#         1.40556454658508,
#         1.4055471420288,
#         1.40553045272827,
#         1.40551912784576,
#         1.40550398826599,
#         1.40548706054687,
#         1.40546822547912,
#         1.40545511245727,
#         1.40543937683105,
#         1.40543484687805,
#         1.40542113780975,
#         1.40540826320648,
#         1.40540480613708,
#         1.40538477897644,
#         1.40537226200103,
#         1.40535151958465,
#         1.40533208847045,
#         1.40532064437866,
#         1.40531682968139,
#         1.40529668331146,
#         1.40528762340545,
#         1.40527820587158,
#         1.40525960922241,
#         1.40525388717651,
#         1.40524172782897,
#         1.40522003173828,
#         1.40520918369293,
#         1.40519452095031,
#         1.40517997741699,
#         1.40517044067382,
#         1.40515518188476,
#         1.4051399230957,
#         1.40513014793395,
#         1.40511870384216,
#         1.40510332584381,
#         1.40509247779846,
#         1.40507972240447,
#         1.40507221221923,
#         1.40505230426788,
#         1.40503764152526,
#         1.40502977371215,
#         1.40500903129577,
#         1.40499997138977,
#         1.40498256683349,
#         1.40496528148651,
#         1.40495634078979,
#         1.40494930744171,
#         1.40493381023406,
#         1.40493094921112,
#         1.40491688251495,
#         1.40489411354064,
#         1.40488588809967,
#         1.40486907958984,
#         1.40484452247619,
#         1.40483820438385,
#         1.40482723712921,
#         1.40481054782867,
#         1.40479826927185,
#         1.40479159355163,
#         1.40477263927459,
#         1.4047622680664,
#         1.40475010871887,
#         1.40473270416259,
#         1.40471625328063,
#         1.40470623970031,
#         1.40468657016754,
#         1.40466976165771,
#         1.4046664237976,
#         1.40465128421783,
#         1.40463471412658,
#         1.40462934970855,
#         1.40460789203643,
#         1.40459752082824,
#         1.40458536148071,
#         1.40456867218017,
#         1.4045512676239,
#         1.40453720092773,
#         1.40451979637145,
#         1.40450775623321,
#         1.40449666976928,
#         1.40448582172393,
#         1.40446138381958,
#         1.40445125102996,
#         1.40444684028625,
#         1.40442848205566,
#     ]

#     routing_period = 300.0

#     for inflow in inflow_list:
#         out, water_elevation = lp_reservoir3.run(inflow, 0.0, routing_period)

#         #print(out)
#         #print(water_elevation)

#     expected_final_outflow = 0.5819599032402039
#     expected_final_water_elevation = 496.2930603027344

#     assert lp_reservoir3 is not None
#     assert expected_final_outflow == pytest.approx(out)
#     assert expected_final_water_elevation == pytest.approx(water_elevation)


# def test_compute_hybrid_run(hybrid_reservoir):
#     """
#     test running a hybrid reservoir
#     """

#     inflow_list = [
#         189.22899,
#         189.27005,
#         189.31049,
#         189.35042,
#         189.38965,
#         189.42819,
#         189.46588,
#         189.50273,
#         189.53859,
#         189.57346,
#         189.60719,
#         189.63979,
#         189.6711,
#         189.7011,
#         189.72968,
#         189.75679,
#         189.7823,
#         189.80617,
#         189.82822,
#         189.84842,
#         189.86653,
#         189.88255,
#         189.89622,
#         189.90752,
#         189.91612,
#         189.922,
#         189.92482,
#         189.92447,
#         189.92067,
#         189.91319,
#         189.90175,
#         189.88611,
#         189.86592,
#         189.84088,
#         189.81064,
#         189.77487,
#         189.73317,
#         189.6852,
#         189.63051,
#         189.56873,
#         189.49939,
#         189.42207,
#         189.33635,
#         189.24176,
#         189.13782,
#         189.02408,
#         188.90009,
#         188.76535,
#         188.61945,
#         188.46188,
#         188.29224,
#         188.11006,
#         187.91493,
#         187.70644,
#         187.48419,
#         187.24779,
#         186.9969,
#         186.73119,
#         186.45035,
#         186.15407,
#         185.84213,
#         185.51424,
#         185.17023,
#         184.80989,
#         184.43312,
#         184.03975,
#         183.62973,
#         183.20296,
#         182.75943,
#         182.29909,
#         181.82205,
#         181.32828,
#         180.81792,
#         80.29099,
#         179.74774,
#         179.1882,
#         178.61267,
#         178.02129,
#         177.41437,
#         176.79207,
#         176.15475,
#         175.50269,
#         174.83627,
#         174.15576,
#         173.46162,
#         172.75417,
#         172.03389,
#         171.3011,
#         170.55634,
#         169.79997,
#         169.03255,
#         168.25441,
#         167.46616,
#         166.66815,
#         165.86099,
#         165.04509,
#         164.22101,
#         163.38913,
#         162.55011,
#         161.70428,
#         160.85229,
#         159.99452,
#         159.13156,
#         158.26382,
#         157.39188,
#         156.51611,
#         155.63715,
#         154.75531,
#         153.8712,
#         152.98517,
#         152.09779,
#         151.2094,
#         150.32057,
#         149.43166,
#         148.54315,
#         147.6554,
#         146.76892,
#         145.88405,
#         145.00128,
#         144.12091,
#     ]

#     routing_period = 300.0

#     for inflow in inflow_list:
#         out, water_elevation = hybrid_reservoir.run(inflow, 0.0, routing_period)
#         #print(out)
#         #print(water_elevation)

#     expected_final_outflow = 13.73367
#     expected_final_water_elevation = 1331.2092285

#     assert hybrid_reservoir is not None
#     assert expected_final_outflow == pytest.approx(out)
#     assert expected_final_water_elevation == pytest.approx(water_elevation)


# def test_compute_rfc_run(rfc_reservoir):
#     """
#     test running a RFC reservoir
#     """

#     inflow_list = [
#         189.22899,
#         189.27005,
#         189.31049,
#         189.35042,
#         189.38965,
#         189.42819,
#         189.46588,
#         189.50273,
#         189.53859,
#         189.57346,
#         189.60719,
#         189.63979,
#         189.6711,
#         189.7011,
#         189.72968,
#         189.75679,
#         189.7823,
#         189.80617,
#         189.82822,
#         189.84842,
#         189.86653,
#         189.88255,
#         189.89622,
#         189.90752,
#         189.91612,
#         189.922,
#         189.92482,
#         189.92447,
#         189.92067,
#         189.91319,
#         189.90175,
#         189.88611,
#         189.86592,
#         189.84088,
#         189.81064,
#         189.77487,
#         189.73317,
#         189.6852,
#         189.63051,
#         189.56873,
#         189.49939,
#         189.42207,
#         189.33635,
#         189.24176,
#         189.13782,
#         189.02408,
#         188.90009,
#         188.76535,
#         188.61945,
#         188.46188,
#         188.29224,
#         188.11006,
#         187.91493,
#         187.70644,
#         187.48419,
#         187.24779,
#         186.9969,
#         186.73119,
#         186.45035,
#         186.15407,
#         185.84213,
#         185.51424,
#         185.17023,
#         184.80989,
#         184.43312,
#         184.03975,
#         183.62973,
#         183.20296,
#         182.75943,
#         182.29909,
#         181.82205,
#         181.32828,
#         180.81792,
#         80.29099,
#         179.74774,
#         179.1882,
#         178.61267,
#         178.02129,
#         177.41437,
#         176.79207,
#     ]

#     routing_period = 3600.0

#     for inflow in inflow_list:
#         out, water_elevation = rfc_reservoir.run(inflow, 0.0, routing_period)
#         #print(out)
#         #print(water_elevation)

#     expected_final_outflow = 3.6
#     expected_final_water_elevation = 1331.436035

#     assert rfc_reservoir is not None
#     assert expected_final_outflow == pytest.approx(out)
#     assert expected_final_water_elevation == pytest.approx(water_elevation)
