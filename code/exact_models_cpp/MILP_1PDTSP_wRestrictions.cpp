#include "MILP_1PDTSP.h"

string itor(int i) { stringstream s; s << i; return s.str(); }

MILP_1PDTSP::MILP_1PDTSP(vector<Node> vNodes, int Q)
{
	try
	{
		GRBEnv env = GRBEnv();
		GRBModel model = GRBModel(env);

		int i, j, k;

		GRBVar y[numberNodes][numberNodes];
		GRBVar x[numberNodes][numberNodes];
		GRBVar z[numberNodes][numberNodes];
        GRBVar xu, xl;


		for (i = 0; i < numberNodes; i++) {
			for (j = 0; j < numberNodes; j++) {
				string s = "y_" + itor(i) + "_" + itor(j);
				string s1 = "x_" + itor(i) + "_" + itor(j);
				string s2 = "z_" + itor(i) + "_" + itor(j);
				y[i][j] = model.addVar(0.0, 1.0, distances[i][j], GRB_BINARY, s);
				x[i][j] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, s1);
				z[i][j] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, s2);
			}
		}
        xu = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, 'x_max');
        xl = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS, 'x_min');
		model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);
		model.update();

		/*
		for (i = 0; i < numberNodes - 1; i++) {
			y[initialS.getPathTSP()[i].get_id()][initialS.getPathTSP()[i + 1].get_id()].set(GRB_DoubleAttr_Start, 1);
		}
		*/

		// 2. Each node is visited exactly once
		for (i = 0; i < numberNodes; i++) {
			GRBLinExpr expr = 0;
			for (j = 0; j < numberNodes; j++) {
				if (i != j)	{
					expr += y[i][j];
				}
			}
			string s = "NodeVisited1_" + itor(i);
			model.addConstr(expr, GRB_EQUAL, 1.0, s);
		}

		// 3. Each node is visited exactly once
		for (i = 0; i < numberNodes; i++) {
			GRBLinExpr expr = 0;
			for (j = 0; j < numberNodes; j++) {
				expr += y[j][i];
			}
			for (j = 0; j < numberNodes; j++) {
				expr -= y[i][j];
			}
			string s = "NodeVisited2_" + itor(i);
			model.addConstr(expr, GRB_EQUAL, 0.0, s);
		}

		// 4. Decreasing order for visited nodes
		for (i = 1; i < numberNodes; i++) {
			GRBLinExpr expr = 0;
			for (k = 0; k < numberNodes; k++){
				expr += z[k][i];
			}
			for (k = 0; k < numberNodes; k++){
				expr -= z[i][k];
			}
			string s = "OrderVisits_" + itor(i);
			model.addConstr(expr, GRB_EQUAL, 1.0, s);
		}

        // No hay 5
        // 5. Order of visites something
        // sum y_0i = n  --> sum z_0i == n
        GRBLinExpr expr = 0;
        for (i = 0; i < numberNodes; i++){
            expr +=  z[0][i];
        }
        string s = "OrderVisits_Node_0";
        model.addConstr(expr, GRB_LESS_EQUAL, numberNodes, s);

		// 6. Order for visited nodes
		for (i = 0; i < numberNodes; i++) {
			for (j = 0; j < numberNodes; j++) {
				GRBLinExpr expr = 0;
				expr += z[i][j];
				expr -= numberNodes * y[i][j];
				string s = "OrderVisits1_" + itor(i) + "_" + itor(j);
				model.addConstr(expr, GRB_LESS_EQUAL, 0.0, s);
			}
		}

        // No hay 7
        // 7. At least one bike has to be carried
        for (i = 0; i < numberNodes; i++) {
			for (j = 0; j < numberNodes; j++) {
				GRBLinExpr expr = 0;
				expr += y[i][j];
				expr -= z[i][j];
				string s = "MinCarriedBikes_" + itor(i) + "_" + itor(j);
				model.addConstr(expr, GRB_LESS_EQUAL, 0.0, s);
			}
		}

		// 8, 9. Demand of each node
		for (i = 0; i < numberNodes; i++) {
			GRBLinExpr expr = 0;
			for (k = 0; k < numberNodes; k++){
				expr += x[k][i];
			}
			for (k = 0; k < numberNodes; k++){
				expr -= x[i][k];
			}
			expr -= vNodes.at(i).getq();
			string s = "Demand_" + itor(i);
			model.addConstr(expr, GRB_EQUAL, 0.0, s);
		}


		// 10, 11. The load of the vehicle can not exceed its capacity
		for (i = 0; i < numberNodes; i++) {
			for (j = 0; j < numberNodes; j++) {
				GRBLinExpr expr = 0;
				expr += x[i][j];
				expr -= Q * y[i][j];
				string s = "VehCapacity_" + itor(i) + "_" + itor(j);
				model.addConstr(expr, GRB_LESS_EQUAL, 0.0, s);
			}
		}

        // No hay 12
        // 12. Upper bound of X
        for (i = 0; i < numberNodes; i++) {
			for (j = 0; j < numberNodes; j++) {
				GRBLinExpr expr = 0;
				expr += x[i][j];
				expr -= xu;
				string s = "MaxCarriedBikes_" + itor(i) + "_" + itor(j);
				model.addConstr(expr, GRB_LESS_EQUAL, 0.0, s);
			}
		}


        // No hay 13
        // 13. Lower bound of X
        for (i = 0; i < numberNodes; i++) {
			for (j = 0; j < numberNodes; j++) {
				GRBLinExpr expr = 0;
				expr += xl;
				expr -= x[i][j];
				string s = "MinCarriedBikes_" + itor(i) + "_" + itor(j);
				model.addConstr(expr, GRB_LESS_EQUAL, 0.0, s);
			}
		}


        // No hay 14
        // 14. Max. capacity
			
        GRBLinExpr expr = 0;
        expr += xu;
        expr -= xl;
        string s = "Max capacity";
        model.addConstr(expr, GRB_LESS_EQUAL, 0.0, s);
			
		

        // Apendixes - Cuts
        // 
		// Cut 1 -- y=0 for arcs(i,i)
		for (i = 0; i < numberNodes; i++) {
			GRBLinExpr expr = 0;
			expr = y[i][i];
			string s = "yvalues_" + itor(i);
			model.addConstr(expr, GRB_EQUAL, 0.0, s);

		}

		model.update();

		// Optimize model

		//model.write("ch130_1.mps");

		model.set(GRB_IntParam_OutputFlag, 1);
		model.set(GRB_DoubleParam_TimeLimit, 3600);


		model.optimize();

		// Save CPU time, objective and gap
		cpuTime = model.get(GRB_DoubleAttr_Runtime);
		objectiveValue = round(model.get(GRB_DoubleAttr_ObjVal));
		gapAbs = model.get(GRB_DoubleParam_MIPGapAbs);

		
		//Save path
		int id = 0;
		do{
			path.push_back(vNodes[id]);
			for (int i = 0; i < numberNodes; i++){
				if (y[id][i].get(GRB_DoubleAttr_X) > 0.9) {
					id = i;
					break;
				}
			}
		} while (id!=0);


		//Build RoutePD object
		vector<int> l, mV, MV;
		int l0 = path[0].getq();
		
		l.push_back(l0);
		mV.push_back(l0);
		MV.push_back(l0);

		int load = l0;
		int m = l0;
		int M = l0;
		int q;
		bool feas = true;

		for (int i = 1; i < numberNodes; i++){
			q = path[i].getq();
			l.push_back(load - q);
			load -= q;
			if (load>M) { M = load; }
			if (load<m) { m = load; }
			mV.push_back(m);
			MV.push_back(M);
			if (M - m > Q) { feas = false; }
		}
		
		vector<int> position = vector<int>(numberNodes);

		for (int i = 0; i <numberNodes; i++) {
			position[path[i].getId()] = i;
		}
	
		route = RoutePD(objectiveValue, path, l, feas, mV, MV, position);
		route.setMaxLoad(MV[MV.size() - 1]);
		route.setMinLoad(mV[MV.size() - 1]);

		
	}
	catch (GRBException e) {
		cout << "Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
	}
	catch (...) {
		cout << "Exception during optimization" << endl;
	}

}


MILP_1PDTSP::MILP_1PDTSP()
{
}




MILP_1PDTSP::~MILP_1PDTSP()
{
}
