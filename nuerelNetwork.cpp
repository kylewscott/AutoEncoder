#include <iostream>
#include <algorithm>
#include <math.h>

using namespace std;

double sigmoid(double x){
    return (1/(1+exp(-x)));
}

double sigmoidPrime(double x){
    return x * (1-x);
}

double relu(double x){
    return max(0.0, x);
}

double reluPrime(double x){
    if(x < 0){
        return 0;
    }
    else{
        return 1;
    }
}