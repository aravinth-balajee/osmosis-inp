#include <iostream>
#include <opencv2/highgui.hpp>
#include <petsc.h>

#include "LinearOsmosis.h"


void cv_mat_to_petsc_vec

    (cv::Mat u,     /* input, image in cv Mat type */ 
    Vec f_petsc)    /* output, image in petsc Vec type */

/* 
    Converts given image from cv Mat data structure to 
    petsc Vec data structure;
    flattens the image by rows
 */

{

    double* array = u.ptr<double>();
    long length = u.cols * u.rows;

    /* create index array */
    int* index = new int[length];
    for(int i = 0; i < length; i++)
        index[i] = i;

    VecSetValues(f_petsc, length, index, array, INSERT_VALUES);
    VecAssemblyBegin(f_petsc);
    VecAssemblyEnd(f_petsc);
    delete[] index;
}

void petsc_vec_to_cv_mat

    (Vec f_petsc,   /* input, image in petsc Vec type */ 
    cv::Mat &u)     /* output, image in cv Mat type */

/* 
    converts the given vector from petsc Vec structure to
    cv Mat data structure;
    ouputs as a flattened image, reshaping required
 */

{   
    long length = u.rows * u.cols;
    double* petsc_vector = new double[length];
    VecGetArray(f_petsc, &petsc_vector);
    cv::Mat pic(1, length, u.type(), petsc_vector);
    u = pic.clone();

    VecRestoreArray(f_petsc, &petsc_vector);
    delete[] petsc_vector;
}

LinearOsmosis::LinearOsmosis

    (cv::Mat u)     /* input, initial image u without boundaries*/

/* 
    configures petsc matrix to hold A(u);
    configures petsc ksp solver with BiCGSTAB;

 */

{

    int nonzeros_per_row = 5;   //  required for configuring petsc matrix
    long i,j;
    sizeX = u.rows;
    sizeY = u.cols;
    cv::Mat k1(sizeY, sizeX, u.type(), cv::Scalar(0.0)); //   holds 1d indices of the 2d image

    // Generate 1d indices for 2d image
    double k = 1;
    for (i=0; i < sizeX; i++)
    for (j=0; j < sizeY; j++)
    {
      k1.at<double>(i, j) = k;
      k = k + 1;
    }

    // Add reflecting boundary conditions to k1 and assign to oneDimIndices
    cv::copyMakeBorder(k1,oneDimIndices,
                        1,1,1,1,cv::BORDER_REFLECT);
    
    // configure petsc and A(u) sparse matrix
    MatCreate(PETSC_COMM_WORLD, &petscA);
    MatSetType(petscA, MATAIJ);  
    MatSetSizes(petscA, PETSC_DECIDE, PETSC_DECIDE, sizeX * sizeY, sizeX * sizeY);
    MatSeqAIJSetPreallocation(petscA, nonzeros_per_row, PETSC_NULL);

    // configure petsc KSP solver with BiCGSTAB solver
    KSPCreate(PETSC_COMM_WORLD, &petscKsp);
    KSPSetType(petscKsp,  KSPBCGS);
    
}

void LinearOsmosis::computeOsmosisWeights

    (double hx,     /* grid size in x direction */
    double hy,      /* grid size in y direction */
    cv::Mat d1,     /* input: drift vector in x direction */
    cv::Mat d2)     /* input: drift vector in y direction */
    
/* 
    updates space discrete matrix petscA = A(u) based on canonical drift
    vectors;
    reflecting boundary conditon is handled
 */

{

    long i,j;
    
    // center and its 4 neighbours
    long curr_pix;
    long left_pix, right_pix; 
    long top_pix, bottom_pix; 

    // weights in A(u)
    double c_px, l_px, r_px, t_px, b_px; 

    // time savers
    double rx, ry, rxx, ryy;

    rx = 1/(2.0 * hx);
    ry = 1/(2.0 * hy);
    rxx = 1/(hx * hx);
    ryy = 1/(hy * hy);

    // update A(u) based on drift vectors
    for(j = 1; j <= sizeY; j++)
    for(i = 1; i <= sizeX; i++){

        /* weight positions */
        curr_pix = oneDimIndices.at<double>(i, j) - 1;
        left_pix = oneDimIndices.at<double>(i, j-1) - 1 ;
        right_pix = oneDimIndices.at<double>(i, j + 1) - 1;
        top_pix = oneDimIndices.at<double>(i -1, j) - 1;
        bottom_pix = oneDimIndices.at<double>(i + 1, j) - 1;


        /* update weights based on space discretisation scheme */
        c_px = - 2.0 * (rxx + ryy) 
            + rx *  (d1.at<double>(i, j -1)  - d1.at<double>(i, j)) 
            + ry * (d2.at<double>(i -1, j) - d2.at<double>(i, j));
        l_px = rxx + rx * d1.at<double>(i, j -1);
        r_px = rxx - rx * d1.at<double>(i, j);
        t_px = ryy + ry * d2.at<double>(i -1, j);
        b_px = ryy - ry * d2.at<double>(i, j);  

        // handle reflective boundary conditons
        if( curr_pix == left_pix)
        c_px = c_px + l_px;

        if( curr_pix == right_pix)
        c_px = c_px + r_px;

        if( curr_pix == top_pix)
        c_px = c_px + t_px;

        if( curr_pix == bottom_pix)
        c_px = c_px + b_px;

        // update weights in petsc matrix
        MatSetValue(petscA, curr_pix, curr_pix, (PetscScalar)c_px, 
                     INSERT_VALUES);

        if( curr_pix != right_pix)
        MatSetValue(petscA, curr_pix, right_pix, (PetscScalar)r_px, 
                    INSERT_VALUES);

        if( curr_pix != left_pix)
        MatSetValue(petscA, curr_pix, left_pix, (PetscScalar)l_px, 
                    INSERT_VALUES);

        if( curr_pix != bottom_pix)
        MatSetValue(petscA, curr_pix, bottom_pix, (PetscScalar)b_px, 
                    INSERT_VALUES);

        if( curr_pix != top_pix)
        MatSetValue(petscA, curr_pix, top_pix, (PetscScalar)t_px, 
                    INSERT_VALUES); 
    }

    MatAssemblyBegin(petscA,MAT_FINAL_ASSEMBLY); 
    MatAssemblyEnd(petscA,MAT_FINAL_ASSEMBLY);
}


void LinearOsmosis::performOsmosis
    (cv:: Mat &u,           /*  input: initial image  */ 
    long time_step_size,    /*  tau, time step size for osmosis evolution */ 
    long max_epoch)         /*  maximum epochs to perform osmosis */

/* 
    computes implicit time scheme for osmosis;
    fully stable for all tau;
    uses BiCGSTAB for fast convergence;
 */

{

    //  updates petscA with implcit scheme weights (I - tau * A(u))
    MatScale(petscA, - time_step_size); MatShift(petscA , 1.0);

    Vec f_petsc, u_petsc;
    long length = u.cols * u.rows;
    cv::Mat f(1, length, u.type());
    
    VecCreate(PETSC_COMM_WORLD, &f_petsc);
    VecSetSizes(f_petsc, PETSC_DECIDE, length);
    VecSetFromOptions(f_petsc);

    KSPSetOperators(petscKsp, petscA, petscA);
    KSPSetFromOptions(petscKsp);

    for (int i=1; i <= max_epoch; i++){

        //  find unknowns by solving equation, 
        // u^k+1 = (I - tau * A(u))^-1 * u^k

        cv_mat_to_petsc_vec(u, f_petsc);
        VecDuplicate(f_petsc, &u_petsc);
        KSPSolve(petscKsp, f_petsc, u_petsc);
        petsc_vec_to_cv_mat(u_petsc, f);
        u = f.reshape(0, sizeX);
    }

    VecDestroy(&u_petsc);
    VecDestroy(&f_petsc);
    MatDestroy(&petscA);
    KSPDestroy(&petscKsp);
   
}