#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <petscmat.h>

#include "LinearOsmosis.h"



/*--------------------------------------------------------------------------*/
/*                                                                          */
/*                  OSMOSIS FOR COMPACT DATA REPRESENTATION                 */
/*                                                                          */
/*                     (Author: Aravinth Balajee, 2/2024)                   */
/*                                                                          */
/*--------------------------------------------------------------------------*/

/* 
    reconstructs the approximation of the image from its drift vectors 
    provided at the edges through osmosis evolution
*/

/*--------------------------------------------------------------------------*/


void read_image

    (std::string& file_path     /* path of the image */, 
    cv::Mat &output_img)        /* cv Mat object, output */

/* 
    reads an image and converts into a grey scale cv Mat object;
    adds one layer of reflective boundary conditions;
    output_img(3,2) refers to pixel location (2,3);
 */

{
    cv::Mat img = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
    if(! img.data)
        {
            std::cout<<"Could not open file:" << file_path << std::endl;
        }
    else 
        // Add Reflecting boundary conditions 
        cv::copyMakeBorder(img,output_img,
                            1,1,1,1,cv::BORDER_REFLECT);
        output_img.convertTo(output_img, CV_64F);
}

void display_image

    (cv::Mat img)   /* image to be displayed */

/* 
    display the image in the cv window after normalizing the given
    image data
 */

{
    cv::Mat output_img;
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
    img.convertTo(output_img, CV_8U); // Convert to 8-bit unsigned integer
    cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Image", output_img);
    cv::waitKey(0);
}


void compute_canonical_drift_vectors

    (cv::Mat org_img,       /* input: original image */
    double hx,              /* grid size in x direction */
    double hy,              /* grid size in y direction */
    cv:: Mat &d1_vector,    /* output, drift vector, x direction */
    cv:: Mat &d2_vector)    /* output, drift vector, y direction */

/* 
    computes the canonical drift vectors at the intergrid locations;
    d = gradient(v)/v;
    d1(i,j) equals to d1(i+1/2, j);
    d2(i,j) equals to d1(i, j+1/2);
 */

{
    long i,j;
    double d1,d2;
    long size_x, size_y;

    // size of the image without boundary layers
    size_x = org_img.rows - 2;
    size_y = org_img.cols - 2;

    // compute canonical drift vectors in x direction
    for(i = 0; i <= size_x; i++)
    for(j = 0; j <= size_y; j++)
        {
            d2 = (2 * ( org_img.at<double>(i+1,j) -  org_img.at<double>(i,j))) /
                (hx * ( org_img.at<double>(i+1,j) + org_img.at<double>(i,j)) );
            d2_vector.at<double>(i, j) = d2;
        }

    // compute canonical drift vectors in y direction
    for(i = 0; i <= size_x; i++)
    for (j = 0; j <= size_y; j++)
        {
            d1 = (2 * ( org_img.at<double>(i,j+1) -  org_img.at<double>(i,j))) /
                (hy * ( org_img.at<double>(i,j+1) + org_img.at<double>(i,j)) );
            d1_vector.at<double>(i, j) = d1;
        }   
}



int main( int argc, char** argv )
{
    PetscInitialize(&argc, &argv, NULL, NULL);

    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] 
                << " <org_img_path> <edge_mask_path> " 
                << "<time_step_size> <max_epochs>" << std::endl;
        return 1;
    }

    // parse parameters from cmd line
    std::string org_img_path = argv[1];
    std::string edge_mask_path = argv[2];
    long time_step_size = std::stol(argv[3]);
    int max_epochs = std::stoi(argv[4]);
    std::string output_img_path = "output/output.pgm";

    // output the provided parameters
    std::cout << "=================================" << std::endl;
    std::cout << "Osmosis reconstruction from edges" << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << "Orginal Image: " << org_img_path << std::endl;
    std::cout << "Edge Mask: " << edge_mask_path << std::endl;
    std::cout << "Time Step Size: " << time_step_size << std::endl;
    std::cout << "Max Epochs: " << max_epochs << std::endl;
    

    int offset = 1; // osmosis requires positive pixels
    double hx = 1.0, hy = 1.0;

    cv::Mat org_img;
    cv::Mat edge_mask_img;
    cv::Mat output_img;

    // read original image, edge mask
    read_image(org_img_path, org_img);
    read_image(edge_mask_path, edge_mask_img);

    // generate initial image with same avg as orginal image
    double average_gray = cv::mean(org_img)[0];
    cv::Mat initial_img(org_img.rows -2, org_img.cols -2,  
                        org_img.type(), cv::Scalar(average_gray));

    // add offset to original, initial image
    org_img.convertTo(org_img, -1, 1, offset);
    initial_img.convertTo(initial_img, -1, 1, offset);
    
    // compute drift vectors, d1, d2
    cv::Mat d1_vector( org_img.size(), org_img.type(), cv::Scalar(0.0));
    cv::Mat d2_vector( org_img.size(), org_img.type(), cv::Scalar(0.0));
    compute_canonical_drift_vectors(org_img, hx, hy, d1_vector, d2_vector);

    // select drift vectors only at the mask, other set to 0 
    edge_mask_img.convertTo(edge_mask_img, CV_8U);
    d1_vector.setTo(0, ~edge_mask_img); 
    d2_vector.setTo(0, ~edge_mask_img);

    // evolve initial image with linear osmosis and drift vectors
    LinearOsmosis linear_osmosis = LinearOsmosis(initial_img);
    linear_osmosis.computeOsmosisWeights(hx, hy, d1_vector, d2_vector);
    linear_osmosis.performOsmosis(initial_img, time_step_size, max_epochs);

    // remove offset before saving the image
    initial_img.convertTo(initial_img, -1, 1,  - offset);
    initial_img.convertTo(initial_img, CV_8U); 
    cv::imwrite(output_img_path, initial_img);
    std::cout << "Reconstructed image, ouput.pgm is saved successfully!! " << std::endl;
    display_image(initial_img);

    PetscFinalize();
    return 0;

}