#include "Node.h"

using namespace std;

Node::Node()
{
}

Node::Node(int id, double x, double y, int dem) {
	Id = id;
	cx = x;
	cy = y;
	q = dem;
}

int Node::getId() {
	return Id;
}

int Node::getq() {
	return q;
}

double Node::getcx() {
	return cx;
}

double Node::getcy() {
	return cy;
}

void Node::setcx(double x) {
	cx = x;
}

void Node::setcy(double y) {
	cy = y;
}

void Node::setq(int dem) {
	q = dem;
}

void Node::setId(int id) {
	Id = id;
}

Node::~Node()
{
}
