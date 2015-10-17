#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>
#include <iterator>

template<typename T = double>
class HMM{
private:
    int states_num;
    int observables_num;
    /*
    t_probs:
        |  S1   |   S2
    -----------------------
    S1  |S1->S1 | S2->S1
    -----------------------
    S2  |S1->S2 | S2->S2
    -----------------------
    */
    std::vector<std::vector<T>> t_probs;
    std::vector<std::vector<T>> e_probs;
public: // forward algorithm
    std::vector<std::vector<T>> alpha;
    std::vector<std::vector<T>> beta;
public: // viterbi algorithm
    std::vector<std::vector<T>> v; // V table
    std::vector<std::vector<int>> backtrack; // backtrack table
public:
    HMM(int s, int o): states_num(s),observables_num(o)
                    , t_probs(states_num, std::vector<T>(states_num, T{}))
                    , e_probs(states_num, std::vector<T>(observables_num, T{}))
                    {}

    HMM(int s, int o, const std::vector<std::vector<T>> & t, const std::vector<std::vector<T>> & e):
                    states_num(s)
                    , observables_num(o)
                    , t_probs(t)
                    , e_probs(e)
                    {}

    // observation is a vector of int, the element
    // corresponds to the column of e_probs
    void forward(const std::vector<int> & observation, const std::vector<T> & initial);

     // return the most likely state sequence
    std::vector<int> viterbi(const std::vector<int> & observation, const std::vector<T> & initial);

    // return the most likely state for each time step
    std::vector<int> backward(const std::vector<int> & observation, const std::vector<T> & initial);

    void print(const std::vector<std::vector<T>> &);
};

template<typename T>
void HMM<T>::forward(const std::vector<int> & observation, const std::vector<T> & initial){
    assert(initial.size() == states_num && "the size of initial is not equal to states_num.");
    auto size = observation.size();
    alpha.resize(size);
    for(auto & e : alpha){
        e.resize(states_num, T{});
    }
    alpha[0] = initial;
    for(unsigned i = 1; i < size; ++i){
        for(unsigned j = 0; j < states_num; ++j){
            auto emission = e_probs[j][observation[i]];
            for(unsigned k = 0; k < states_num; ++k){
                alpha[i][j] += alpha[i-1][k]*t_probs[j][k];
            }
            alpha[i][j] *= emission;
        }
    }
    std::cout<<"------- alpha table --------"<<std::endl;
    print(alpha);
}

template<typename T>
std::vector<int> HMM<T>::viterbi(const std::vector<int> & observation, const std::vector<T> & initial){
    assert(initial.size() == states_num && "the size of initial is not equal to states_num.");
    auto size = observation.size();
    v.resize(size);
    backtrack.resize(size);
    for(unsigned i = 0; i < size; ++i){
        v[i].resize(states_num);
        backtrack[i].resize(states_num, 0);
    }
    unsigned i;

    v[0] = initial;
    std::vector<T> tmp(states_num);

    for(i = 1; i < size; ++i){
        for(unsigned j = 0; j < states_num; ++j){
            auto emission = e_probs[j][observation[i]];
            std::transform(v[i-1].begin(), v[i-1].end(), t_probs[j].begin(), tmp.begin(), std::multiplies<T>());
            auto max_it = std::max_element(tmp.begin(), tmp.end());
            auto max_index = std::distance(tmp.begin(), max_it);

            backtrack[i][j] = max_index;
            v[i][j] = emission * v[i-1][max_index] * t_probs[j][max_index];
        }
    }

    std::cout<<"------- v table --------"<<std::endl;
    for(auto & e : v){
        for(auto i : e){
            std::cout<<i<<"  ";
        }
        std::cout<<std::endl;
    }
    //std::cout<<"------------------------"<<std::endl;

    auto max_it = std::max_element(v[size-1].begin(), v[size-1].end());
    auto max_index = std::distance(v[size-1].begin(), max_it);
    std::vector<int> res(size);
    res[size-1] = max_index;
    for(int j = size-1; j > 0; --j){
        max_index = res[j];
        res[j-1] = backtrack[j][max_index];
    }

    std::cout<<"------------- moste likely state sequence -----------"<<std::endl;
    for(auto e : res){
        std::cout<<e<<"  ";
    }
    std::cout<<std::endl;
    //std::cout<<"-----------------------------------------------------"<<std::endl;
    return res;
}


template<typename T>
std::vector<int> HMM<T>::backward(const std::vector<int> & observation, const std::vector<T> & initial){

    assert(initial.size() == states_num && "the size of initial is not equal to states_num.");
    auto size = observation.size();
    beta.resize(size);
    for(auto & e : beta){
        e.resize(states_num, T{});
    }
    beta.back() = initial;
    for(int i = size-2; i >= 0 ; --i){
        for(unsigned j = 0; j < states_num; ++j){
            for(unsigned k = 0; k < states_num; ++k){
                beta[i][j] += beta[i+1][k] * t_probs[k][j] * e_probs[k][observation[i+1]];
            }
        }
    }

    std::cout<<"------- beta table --------"<<std::endl;
    print(beta);

    std::vector<int> res(size, 0);
    for(int i = 0; i < size; ++i){
        double tmp1 = 0.0;
        double tmp2 = 0.0;
        double sum = 0.0;

        for(int j = 0; j < states_num; ++j){
            sum += alpha[i][j]*beta[i][j];
        }

        for(int j = 0; j < states_num; ++j){
            tmp2 = alpha[i][j]*beta[i][j];
            //std::cout<<" ** "<<tmp2/sum<<std::endl;
            if(tmp2 > tmp1){
                res[i] = j;
                tmp1 = tmp2;
            }
        }
    }
    std::cout<<"-------- most likely state at each time step --------"<<std::endl;
    for(auto e : res){
        std::cout<<e<<"  ";
    }
    std::cout<<std::endl;
    //std::cout<<"-----------------------------------------------------"<<std::endl;

    return res;
}

template<typename T>
void HMM<T>::print(const std::vector<std::vector<T>> & p){
    for (int i = 0; i < p.size(); ++i)
    {
        std::cout<<i;
        for (int j = 0; j < p[0].size(); ++j)
        {
            std::cout<<"   "<<p[i][j];
        }
        std::cout<<std::endl;
    }
}