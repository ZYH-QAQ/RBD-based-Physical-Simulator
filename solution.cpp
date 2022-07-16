#include <stdio.h>
#include <omp.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
double* matrixMultiply(double* result,double* m1,double* m2)//矩阵乘法
{
    #pragma omp parallel num_threads(9)
    {
        int id=omp_get_thread_num();
        int i=id/3;
        int j=id%3;
        result[id]=0;
        for(int k=0;k<3;k++)
        {
            result[id]+=m1[i*3+k]*m2[k*3+j];
        }
    }
    return result;
}
double* mvMultiply(double* result,double* m,double* v)
{
    #pragma omp parallel num_threads(3)
    {
        int id=omp_get_thread_num();
        result[id]=m[id*3]*v[0]+m[id*3+1]*v[1]+m[id*3+2]*v[2];
    }
    return result;
}
double* vmMultiply(double* result,double* m,double* v)
{
    #pragma omp parallel num_threads(3)
    {
        int id=omp_get_thread_num();
        result[id]=m[id]*v[0]+m[id+3]*v[1]+m[id+6]*v[2];
    }
    return result;
}

double* vecToLeeAlgebra(double* result,double* w)
{
    result[0]=result[4]=result[8]=0;
    result[1]=-(result[3]=w[2]);
    result[6]=-(result[2]=w[1]);
    result[5]=-(result[7]=w[0]);
    return result;
}
double* cross(double* result,double* v1,double* v2)
{
    result[0]=v1[1]*v2[2]-v2[1]*v1[2];
    result[1]=-v1[0]*v2[2]+v2[0]*v1[2];
    result[2]=v1[0]*v2[1]-v2[0]*v1[1];
    return result;
}
double dot(double* v1,double* v2)
{
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}
double absV(double* v)
{
    return sqrt(dot(v,v));
}
double* transMatrix(double* result,double* m)
{
    #pragma omp parallel num_threads(9)
    {
        int id=omp_get_thread_num();
        int i=id/3;
        int j=id%3;
        result[id]=m[3*j+i];
    }
    return result;    
}
double* normalizeVextor(double* result,double* v)
{
    double mode=absV(v);
    result[0]=v[0]/mode;
    result[1]=v[1]/mode;
    result[2]=v[2]/mode;
    return result;
}
void normalizeMatrix(double* m)
{
    double v1[3]={m[0],m[3],m[6]};
    double v1n[3];
    normalizeVextor(v1n,v1);
    double v3[3]={m[2],m[5],m[8]};
    double v2m[3];
    cross(v2m,v3,v1);
    double v2n[3];
    normalizeVextor(v2n,v2m);
    double v3n[3];
    cross(v3n,v1n,v2n);
    m[0]=v1n[0];m[3]=v1n[1];m[6]=v1n[2];
    m[1]=v2n[0];m[4]=v2n[1];m[7]=v2n[2];
    m[2]=v3n[0];m[5]=v3n[1];m[8]=v3n[2];
}
double* vecAdd(double* result,double* v1,double* v2)
{
    result[0]=v1[0]+v2[0];
    result[1]=v1[1]+v2[1];
    result[2]=v1[2]+v2[2];
    return result;
}
void vecAddUpdate(double* v1,double k,double* v2)//v1:=v1+k*v2
{
    for(int i=0;i<3;i++)
    {
        v1[i]=v1[i]+k*v2[i];
    }
}
void matrixAddUpdate(double* m1,double k,double* m2)
{
    #pragma omp parallel num_threads(9)
    {
        int id=omp_get_thread_num();
        m1[id]=m1[id]+k*m2[id];
    }
}
void matrixMultUpdate(double* m1,double k){
    #pragma omp parallel num_threads(9)
    {
        int id=omp_get_thread_num();
        m1[id]=k*m1[id];
    }
}
double* vecMinus(double* result,double* v1,double* v2)
{
    result[0]=v1[0]-v2[0];
    result[1]=v1[1]-v2[1];
    result[2]=v1[2]-v2[2];
    return result;
}
double* svTimes(double* result,double s,double* v)
{
    for(int i=0;i<3;i++)
    {
        result[i]=s*v[i];
    }
    return result;
}
void printMatrix(double* m)
{
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            printf("%.2f ",*(m+3*i+j));
        }
        printf("\n");
    }
}
void printVector(double* v)
{
    for(int i=0;i<3;i++)
    {
        printf("%.2f ",v[i]);
    }
    printf("\n");
}
//Global variables
double g=9.8,r=0.3,k=500,d=0.8,m=5;
double I[9]={5.0/12*m*r*r,0,0,0,5.0/12*m*r*r,0,0,0,2.0/3*m*r*r};
double _I[9]={12.0/(5*m*r*r),0,0,0,12.0/(5*m*r*r),0,0,0,3.0/(2*m*r*r)};
double r1[3]={};
double rc[3]={-0.12910922,0.10808038,-0.94932212};
double vc[3]={0,0,0};
double r021[3]={r,0,0.5*r};
double r022[3]={-0.5*r,-sqrt(3)/2*r,0.5*r};
double r023[3]={-0.5*r,sqrt(3)/2*r,0.5*r};
double r21[3]={rc[0]+r,rc[1],rc[2]+0.5*r};
double r22[3]={rc[0]-0.5*r,rc[1]-sqrt(3)/2*r,rc[2]+0.5*r};
double r23[3]={rc[0]-0.5*r,rc[1]+sqrt(3)/2*r,rc[2]+0.5*r};
double dt=0.00001;
double R[9]={0.96236609,0.02924754,-0.27017790,0.03122722,0.97570462,0.21685331,0.26995625,-0.21712918,0.93807172};
double dR[9]={};
double omega[3]={0,0,0};
double f1[3]={},f2[3]={},f3[3]={},f[3]={},G[3]={0,0,-m*g};
double M[3]={};


int main()
{
    FILE *file;
    file=fopen("result.txt","w");
    int count=0;
    double countTime=0;
    //ini r21
    double temp[3];
    vmMultiply(temp,R,r021);
    vecAdd(r21,rc,temp);
    //update r22
    vmMultiply(temp,R,r022);
    vecAdd(r22,rc,temp);
    //update r23
    vmMultiply(temp,R,r023);
    vecAdd(r23,rc,temp);
    for(double time=0;time<10.001;time+=dt)
    {
        fprintf(stderr,"\rProceeding %5.2f%%",time/10.001*100);
        count++;
        countTime+=dt;
        if(count%10==0)
        {
            normalizeMatrix(R);
        }
        if(countTime>0.01)
        {
            countTime-=0.01;
            fprintf(file,"%.8f\n",R[8]);
        }
        //calculate M f
        {
            double M1[3],M2[3],M3[3];
            //calculate f1
            {
                double r21_1[3];
                double rc_21[3];
                vecMinus(r21_1,r1,r21);
                vecMinus(rc_21,r21,rc);
                double k_1=1-d/absV(r21_1);
                svTimes(f1,k_1*k,r21_1);
                cross(M1,rc_21,f1);
            }
            //calculate f2
            {
                double r22_1[3];
                double rc_22[3];
                vecMinus(r22_1,r1,r22);
                vecMinus(rc_22,r22,rc);
                double k_1=1-d/absV(r22_1);
                svTimes(f2,k_1*k,r22_1);
                cross(M2,rc_22,f2);
            }
            //calculate f3
            {
                double r23_1[3];
                double rc_23[3];
                vecMinus(r23_1,r1,r23);
                vecMinus(rc_23,r23,rc);
                double k_1=1-d/absV(r23_1);
                svTimes(f3,k_1*k,r23_1);
                cross(M3,rc_23,f3);
            }
            double temp[3];
            vecAdd(temp,M1,M2);
            vecAdd(M,M3,temp);
            vecAdd(temp,f1,f2);
            vecAdd(f,temp,f3);
            vecAddUpdate(f,1,G);
        }
        //update transition
        {
            //first update speed
            {
                vecAddUpdate(vc,dt/m,f);
            }
            //second update position
            {
                vecAddUpdate(rc,dt,vc);
            }
        }
        //update rotation
        {
            //first update speed
            {
                double dRt[9],Rt[9];
                transMatrix(dRt,dR);
                transMatrix(Rt,R);
                //calculate (dR/dt)^t*I*R*omega
                double temp1[9],temp2[9];
                double result1[3];
                matrixMultiply(temp1,dRt,I);
                matrixMultiply(temp2,temp1,R);
                mvMultiply(result1,temp2,omega);
                //calculate R^t*I*(dR/dt)*omega
                double result2[3];
                matrixMultiply(temp1,Rt,I);
                matrixMultiply(temp2,temp1,dR);
                mvMultiply(result2,temp2,omega);
                //calculate M:=Rt*I*R*d_omega/dt
                vecAddUpdate(M,-1,result1);
                vecAddUpdate(M,-1,result2);
                //calculate d_omega/dt
                double d_omega[3];
                matrixMultiply(temp1,Rt,_I);
                matrixMultiply(temp2,temp1,R);
                mvMultiply(d_omega,temp2,M);
                //update omega
                vecAddUpdate(omega,dt,d_omega);
                //calculate dR/dt
                double omegaLee[9];
                vecToLeeAlgebra(omegaLee, omega);
                matrixMultiply(dR,R,omegaLee);
                matrixMultUpdate(dR,-1);
            }
            //second update posture
            {
                matrixAddUpdate(R,dt,dR);
                //update r21
                double temp[3];
                vmMultiply(temp,R,r021);
                vecAdd(r21,rc,temp);
                //update r22
                vmMultiply(temp,R,r022);
                vecAdd(r22,rc,temp);
                //update r23
                vmMultiply(temp,R,r023);
                vecAdd(r23,rc,temp);
            }
        }

    }

}

