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
        //printf("id=%d ref=%f\n",id,result[id]);
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
double rc[3]={0,0,-0.9};
double vc[3]={0,0,0};
double r021[3]={r,0,0.5*r};
double r022[3]={-0.5*r,-sqrt(3)/2*r,0.5*r};
double r023[3]={-0.5*r,sqrt(3)/2*r,0.5*r};
//double r03[3]={0.766044431*r,-0.6427876097*r,0.5*r};
double r03[3]={r,0,0.5*r};
double r21[3]={rc[0]+r,rc[1],rc[2]+0.5*r};
double r22[3]={rc[0]-0.5*r,rc[1]-sqrt(3)/2*r,rc[2]+0.5*r};
double r23[3]={rc[0]-0.5*r,rc[1]+sqrt(3)/2*r,rc[2]+0.5*r};
double r3[3]={};
double dt=0.001;
double R[9]={1,0,0,0,1,0,0,0,1};
double dR[9]={};
double omega[3]={0,0,0};
double f1[3]={},f2[3]={},f3[3]={},f[3]={},G[3]={0,0,-m*g};
double M[3]={};
double fa[3]={0,0,-50};
double M1[3],M2[3],M3[3],Ma[3];
double d_omega[3];
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
    //initialize r3
    vmMultiply(temp,R,r03);
    vecAdd(r3,rc,temp);
    for(double time=0;time<60.01;time+=dt)
    {
        fprintf(stderr,"\rProceeding %5.2f%%",time/60.01*100);
        count++;
        countTime+=dt;
        if(count%10==0)
        {
            normalizeMatrix(R);
        }
        if(countTime>0.1)
        {
            countTime-=0.1;
            fprintf(file,"time=%5.2f rc=(%.8f,%.8f,%.8f) n=(%5.3f,%5.3f,%.8f) M=(%.4f,%.4f,%.4f) f=(%.4f,%.4f,%.4f)\r\n",time,rc[0],rc[1],rc[2],R[6],R[7],R[8],M[0],M[1],M[2],f[0],f[1],f[2]);
            //fprintf(file,"      M1=(%5.3f,%5.3f,%5.3f) M2=(%5.3f,%5.3f,%5.3f) M3=(%5.3f,%5.3f,%5.3f) Ma=(%5.3f,%5.3f,%5.3f)\r\n",M1[0],M1[1],M1[2],M2[0],M2[1],M2[2],M3[0],M3[1],M3[2],Ma[0],Ma[1],Ma[2]);
            fprintf(file,"      r1=(%5.3f,%5.3f,%5.3f) r2=(%5.3f,%5.3f,%5.3f) r3=(%5.3f,%5.3f,%5.3f) ra=(%5.3f,%5.3f,%5.3f)\r\n",r21[0],r21[1],r21[2],r22[0],r22[1],r22[2],r23[0],r23[1],r23[2],r3[0],r3[1],r3[2]);
            fprintf(file,"              d_omega=(%5.3f,%5.3f,%5.3f)\r\n",d_omega[0],d_omega[1],d_omega[2]);
            fprintf(file,"                      rac=(%5.3f,%5.3f,%5.3f)\r\n",r3[0]-rc[0],r3[1]-rc[1],r3[2]-rc[2]);
            fprintf(file,"                      %.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\r\n",R[0],R[1],R[2],R[3],R[4],R[5],R[6],R[7],R[8]);
        }
        //calculate M f
        {
            
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
            //calculate Ma
            {
                double rc_3[3];
                vecMinus(rc_3,r3,rc);
                cross(Ma,rc_3,fa);
            }
            double temp[3];
            vecAdd(temp,M1,M2);
            vecAdd(M,M3,temp);
            vecAddUpdate(M,1,Ma);
            vecAdd(temp,f1,f2);
            vecAdd(f,temp,f3);
            vecAddUpdate(f,1,G);
            vecAddUpdate(f,1,fa);
        }
        //update transition
        {
            //first update speed 0.96->0.01
            {
                vecAddUpdate(vc,dt/m,f);
                vc[0]=vc[0]*0.97;
                vc[1]=vc[1]*0.97;
                vc[2]=vc[2]*0.97;
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
                
                matrixMultiply(temp1,Rt,_I);
                matrixMultiply(temp2,temp1,R);
                mvMultiply(d_omega,temp2,M);
                //update omega 0.99->0.01
                vecAddUpdate(omega,dt,d_omega);
                omega[0]=omega[0]*0.99;
                omega[1]=omega[1]*0.99;
                omega[2]=omega[2]*0.99;
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
                //update r3
                vmMultiply(temp,R,r03);
                vecAdd(r3,rc,temp);
            }
        }

    }

}

