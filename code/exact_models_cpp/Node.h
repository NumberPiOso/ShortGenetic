#pragma once

#include <iostream>
#include <string.h>
#include <stdio.h>
#include <fstream>
#include <cstring>
#include <math.h>
#include <stdlib.h>


using namespace std;

class Node
{
	int Id;
	double cx;
	double cy;
	int q;

public:

	void setId(int id);
	void setcx(double x);
	void setcy(double y);
	void setq(int dem);

	int getId();
	int getq();
	double getcx();
	double getcy();

	Node();
	Node(int id, double x, double y, int dem);
	~Node();
};

