
#ifndef Option_h
#define Option_h



class Option{
public:
    double T,K,sigma,r;
    Option(double T,double K,double sigma,double r);
    double payOff(double s); //European call option
    double PriceByBSFormula(double S0);
    
};

#endif /* Option_h */
