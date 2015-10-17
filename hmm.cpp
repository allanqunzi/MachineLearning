#include <unordered_map>
#include <string>
#include "hmm.h"

using PROB_TYPE = double;

int main(int argc, char const *argv[])
{
    std::vector<std::vector<PROB_TYPE>> transition{
        {0.8, 0.2},
        {0.2, 0.8},
    };
    std::vector<std::vector<PROB_TYPE>> emission{
        {0.3, 0.2, 0.2, 0.3},
        {0.1, 0.4, 0.4, 0.1},
    };
    std::unordered_map<char, int>base_map{
        {'A', 0},
        {'T', 1},
        {'C', 2},
        {'G', 3}
    };
    std::string seq{"CGTCAG"};
    std::vector<int> observation{2, 3, 1, 2, 0, 3};
    HMM<> hmm_dna(transition.size(), emission.size(), transition, emission);

    // question 1
    std::vector<PROB_TYPE> initial_alpah{0.5*0.2, 0.5*0.4};
    hmm_dna.forward(observation, initial_alpah);

    auto res1 = 0.0;
    for(auto e : hmm_dna.alpha.back()){
        res1 += e;
    }
    std::cout<<"res for q(1): "<<res1<<std::endl;

    // question 2
    auto res2 = hmm_dna.viterbi(observation, initial_alpah);

    // question 3
    std::vector<PROB_TYPE> initial_beta{1.0, 1.0};
    auto res3 = hmm_dna.backward(observation, initial_beta);

    return 0;
}