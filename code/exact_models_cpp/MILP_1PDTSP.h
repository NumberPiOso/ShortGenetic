#pragma once

#include "MathModel.h"

using namespace std;

class MILP_1PDTSP:
	public MathModel
{
public:




	MILP_1PDTSP();
	//MILP_1PDTSP(vector<Node>, RoutePD);
	MILP_1PDTSP(vector<Node>, int Q);

	~MILP_1PDTSP();
};

