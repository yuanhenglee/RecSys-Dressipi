#include<iostream>
#include<map>
#include<set>
#include<fstream>
#include<string>
using namespace std;
int main(){
    ifstream input;
    input.open("./dataset/train_sessions.csv");
    
    ofstream output;
    output.open("./dataset/sessionClustering.csv");
    string line;
    int comaAcc = 0;
    string itemID = "";
    string sessionID = "";
    map<int, set<int>> dict;
    while(getline(input, line)){
        if(line != "session_id,item_id,date"){
            for(int i = 0; i<line.size(); i++){
                if(line[i] == ','){
                    comaAcc++;
                }else{
                    if(comaAcc == 2){
                       
                        dict[stoi(sessionID)].insert(stoi(itemID));
                        break;
                    }
                    if(comaAcc == 1){
                        itemID += line[i];
                    }
                    if(comaAcc == 0){
                        sessionID += line[i];
                    }
                }
            }
            comaAcc = 0;
            itemID = "";
            sessionID = "";
        }
        
    }
    output<<"size,sessionID,itemID"<<endl;
    for(auto s :dict){
        output<<s.second.size();
        output<<","<<s.first;
        for(auto e : s.second){
            output<<","<<e;
        }
        output<<endl;
    }
    return 0;
}