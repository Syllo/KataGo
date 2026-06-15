#include "../tests/tests.h"
#include "../neuralnet/nninterface.h"

#include <cmath>

using namespace std;

static bool approxEqual(float x, float y, bool useFP16) {
  float tolerance;
  if(useFP16)
    tolerance = 0.03f * std::max(std::fabs(x),std::max(std::fabs(y),3.0f));
  else
    tolerance = 0.0001f * std::max(std::fabs(x),std::max(std::fabs(y),1.0f));
  return std::fabs(x - y) < tolerance;
}

static void checkApproxEqual(
  const string& label,
  const vector<float>& vec, const vector<float>& expected, int nSize, int cSize, int ySize, int xSize, bool useFP16,
  const char* file, const char* func, int line
) {
  int cyxSize = cSize * ySize * xSize;
  int yxSize = ySize * xSize;

  int totalSize = nSize * cSize * ySize * xSize;
  if (expected.size() < totalSize) {
    cout << "Size mismatch: expected = " << expected.size() << " totalSize = " << totalSize << endl;
    return;
  }
  if (vec.size() < totalSize) {
    cout << "Size mismatch: vec = " << vec.size() << " totalSize = " << totalSize << endl;
    return;
  }

  bool mismatch = false;
  for(int n = 0; n < nSize; n++) {
    for(int c = 0; c < cSize; c++) {
      for(int y = 0; y < ySize; y++) {
        for(int x = 0; x < xSize; x++) {
          int i = n * cyxSize + c * yxSize + y * xSize + x;
          if(!approxEqual(vec[i],expected[i],useFP16) && !mismatch) {
            mismatch = true;
            cout << "File " << file << " func " << func << " line " << line << endl;
            cout << label << endl;
            cout << "Test failed at n c y x = " << n << " " << c << " " << y << " " << x << endl;
          }
        }
      }
    }
  }
  if(mismatch) {
    cout << "==========" << endl;
    cout << "Actual" << endl;
    cout << "==========" << endl;
    for(int n = 0; n < nSize; n++) {
      for(int c = 0; c < cSize; c++) {
        for(int y = 0; y < ySize; y++) {
          for(int x = 0; x < xSize; x++) {
            int i = n * cyxSize + c * yxSize + y * xSize + x;
            cout << Global::strprintf("%.5g, ",vec[i]);
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << "-------" << endl;
    }
    cout << "==========" << endl;
    cout << "Expected" << endl;
    cout << "==========" << endl;
    for(int n = 0; n < nSize; n++) {
      for(int c = 0; c < cSize; c++) {
        for(int y = 0; y < ySize; y++) {
          for(int x = 0; x < xSize; x++) {
            int i = n * cyxSize + c * yxSize + y * xSize + x;
            cout << Global::strprintf("%.5g, ",expected[i]);
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << "-------" << endl;
    }
  }
}
#define CHECK_APPROX_EQUAL(label,vec,expected,n,c,h,w,useFP16) (checkApproxEqual((label),(vec),(expected),(n),(c),(h),(w),(useFP16),__FILE__,#vec,__LINE__))


static vector<float> NCHWtoNHWC(const vector<float>& vec, int nSize, int cSize, int ySize, int xSize) {
  vector<float> ret(vec.size());
  int cyxSize = cSize * ySize * xSize;
  int yxSize = ySize * xSize;
  int xcSize = xSize * cSize;
  for(int n = 0; n < nSize; n++) {
    for(int c = 0; c < cSize; c++) {
      for(int y = 0; y < ySize; y++) {
        for(int x = 0; x < xSize; x++) {
          ret[n * cyxSize + y * xcSize + x * cSize + c] = vec[n * cyxSize + c * yxSize + y * xSize + x];
        }
      }
    }
  }
  return ret;
}


static void testConvLayer(int64_t& numTestsRun) {

  auto testConfigurations = [&](
    const string& label,
    int batchSize, int nnXLen, int nnYLen,
    const ConvLayerDesc& desc, const vector<float>& input, const vector<float>& expected
  ) {
    for(int useNHWC = 0; useNHWC <= 1; useNHWC++) {
      for(int useFP16 = 0; useFP16 <= 1; useFP16++) {
        vector<float> inputThisLoop = useNHWC ? NCHWtoNHWC(input,batchSize,desc.inChannels,nnYLen,nnXLen) : input;
        vector<float> expectedThisLoop = useNHWC ? NCHWtoNHWC(expected,batchSize,desc.outChannels,nnYLen,nnXLen) : expected;

        vector<float> outputThisLoop;
        bool supported = NeuralNet::testEvaluateConv(
          &desc,batchSize,nnXLen,nnYLen,useFP16,useNHWC,inputThisLoop,outputThisLoop
        );

        if(supported) {
          numTestsRun += 1;
          string subLabel = label + Global::strprintf(" useNHWC %d useFP16 %d", useNHWC, useFP16);
          if(useNHWC)
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,nnYLen,nnXLen,desc.outChannels,useFP16);
          else
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,desc.outChannels,nnYLen,nnXLen,useFP16);
        }
      }
    }
  };

  {
    int batchSize = 2;
    int inChannels = 2;
    int nnYLen = 3;
    int nnXLen = 4;

    //NCHW
    vector<float> input({
      5,5,4,4,
      5,5,4,4,
      1,1,8,8,

      0,1,2,3,
      3,4,5,6,
      8,7,6,5,

      0,1,0,2,
      3,0,4,0,
      0,5,0,6,

      1,0,0,2,
      0,2,2,0,
      0,2,2,0,
    });

    {
      string label("1x1 convolution");

      //oc,ic,y,x
      vector<float> convWeights({
          0.0f,1.0f,
          1.0f,-1.0f,
          10.0f,0.1f,
      });
      //NCHW
      vector<float> expected({
        0.0f, 1.0f, 2.0f, 3.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        8.0f, 7.0f, 6.0f, 5.0f,

        5.0f, 4.0f, 2.0f, 1.0f,
        2.0f, 1.0f, -1.0f, -2.0f,
        -7.0f, -6.0f, 2.0f, 3.0f,

        50.0f, 50.1f, 40.2f, 40.3f,
        50.3f, 50.4f, 40.5f, 40.6f,
        10.8f, 10.7f, 80.6f, 80.5f,

        1.0f, 0.0f, 0.0f, 2.0f,
        0.0f, 2.0f, 2.0f, 0.0f,
        0.0f, 2.0f, 2.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 0.0f,
        3.0f, -2.0f, 2.0f, 0.0f,
        0.0f, 3.0f, -2.0f, 6.0f,

        0.1f, 10.0f, 0.0f, 20.2f,
        30.0f, 0.2f, 40.2f, 0.0f,
        0.0f, 50.2f, 0.2f, 60.0f,
      });

      ConvLayerDesc desc;
      desc.convYSize = 1;
      desc.convXSize = 1;
      desc.inChannels = inChannels;
      desc.outChannels = 3;
      desc.dilationY = 1;
      desc.dilationX = 1;
      desc.weights = convWeights;

      testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,expected);
    }

    {
      string label("3x3 convolution");

      //oc,ic,y,x
      vector<float> convWeights({
          1,0,0,
          0,0,0,
          0,0,0,

          0,0,0,
          0,0,0,
          0,0,0,

          0,0,0,
          0,0,1,
          0,1,0,

          0,0,0,
          0,-1,0,
          0,0,0,

          0,0,0,
          0,1,0,
          0,0,0,

          0,0,0,
          0,0,0,
          0,0,2,
      });
      //NCHW
      vector<float> expected({
        0, 0, 0, 0,
        0, 5, 5, 4,
        0, 5, 5, 4,

        10, 8, 6, 1,
        3, 1, 7, 2,
        -7, 1, 2, -5,

        13, 15, 16, 4,
        19, 17, 14, 4,
        1, 1, 8, 8,

        0, 0, 0, 0,
        0, 0, 1, 0,
        0, 3, 0, 4,

        3, 0, 6, -2,
        0, 7, -2, 6,
        5, -2, 4, 0,

        4, 5, 0, 2,
        7, 4, 4, 0,
        0, 5, 0, 6,
      });

      ConvLayerDesc desc;
      desc.convYSize = 3;
      desc.convXSize = 3;
      desc.inChannels = inChannels;
      desc.outChannels = 3;
      desc.dilationY = 1;
      desc.dilationX = 1;
      desc.weights = convWeights;

      testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,expected);
    }

    {
      string label("5x5 convolution");

      //oc,ic,y,x
      vector<float> convWeights({
          0,0,0,0,1,
          0,0,0,1,0,
          0,0,1,0,0,
          0,0,0,0,0,
          0,0,0,0,0,

          0,0,0,0,0,
          0,0,0,0,0,
          0,0,1,0,0,
          0,1,0,0,0,
          1,0,0,0,0,

          0,0,0,0,0,
          0,0,0,0,0,
          0,0,0,0,0,
          0,0,0,0,0,
          0,0,0,0,2,

          0,0,0,0,0,
          0,0,1,0,0,
          2,0,0,0,0,
          0,0,0,0,0,
          0,0,0,0,0,
      });

      //NCHW
      vector<float> expected({
        5, 9,18,19,
       13,21,20,16,
       18,16,18,13,

       16,16, 0, 2,
        0, 1, 8,11,
        3, 4,21,20,

        1, 1, 2, 8,
        4, 2,10, 2,
        0,13, 2, 6,

        0,12, 2, 0,
        1, 0, 0, 6,
        0, 2, 2, 4,
      });

      ConvLayerDesc desc;
      desc.convYSize = 5;
      desc.convXSize = 5;
      desc.inChannels = inChannels;
      desc.outChannels = 2;
      desc.dilationY = 1;
      desc.dilationX = 1;
      desc.weights = convWeights;

      testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,expected);
    }

    // This shape is deliberately eligible for the Vulkan NHWC vec8 implicit
    // GEMM path (M=64, N=64, C=64). It remains a normal reference test when
    // the opt-in environment variable is absent, and exercises both layout
    // boundary shaders when it is present.
    {
      string label("3x3 convolution coopmat NHWC boundary");
      const int largeBatchSize = 1;
      const int largeInChannels = 64;
      const int largeOutChannels = 64;
      const int largeNnXLen = 8;
      const int largeNnYLen = 8;
      vector<float> largeInput((size_t)largeBatchSize * largeInChannels * largeNnXLen * largeNnYLen);
      vector<float> largeWeights((size_t)largeOutChannels * largeInChannels * 3 * 3);
      for(size_t i = 0; i < largeInput.size(); i++)
        largeInput[i] = (float)((int)(i % 23) - 11) * 0.03125f;
      for(size_t i = 0; i < largeWeights.size(); i++)
        largeWeights[i] = (float)((int)(i % 19) - 9) * 0.015625f;
      vector<float> largeExpected((size_t)largeBatchSize * largeOutChannels * largeNnXLen * largeNnYLen, 0.0f);
      for(int oc = 0; oc < largeOutChannels; oc++) {
        for(int y = 0; y < largeNnYLen; y++) {
          for(int x = 0; x < largeNnXLen; x++) {
            float sum = 0.0f;
            for(int ic = 0; ic < largeInChannels; ic++) {
              for(int fy = 0; fy < 3; fy++) {
                const int iy = y + fy - 1;
                if(iy < 0 || iy >= largeNnYLen)
                  continue;
                for(int fx = 0; fx < 3; fx++) {
                  const int ix = x + fx - 1;
                  if(ix < 0 || ix >= largeNnXLen)
                    continue;
                  sum += largeInput[(size_t)ic * largeNnXLen * largeNnYLen + iy * largeNnXLen + ix] *
                         largeWeights[((size_t)(oc * largeInChannels + ic) * 3 + fy) * 3 + fx];
                }
              }
            }
            largeExpected[(size_t)oc * largeNnXLen * largeNnYLen + y * largeNnXLen + x] = sum;
          }
        }
      }
      ConvLayerDesc largeDesc;
      largeDesc.convYSize = 3;
      largeDesc.convXSize = 3;
      largeDesc.inChannels = largeInChannels;
      largeDesc.outChannels = largeOutChannels;
      largeDesc.dilationY = 1;
      largeDesc.dilationX = 1;
      largeDesc.weights = largeWeights;
      testConfigurations(label, largeBatchSize, largeNnXLen, largeNnYLen, largeDesc, largeInput, largeExpected);
    }

    // Keep a 5x5 case at the same alignment boundary. This reaches the
    // CONV_SIZE=5 pipeline specialization when native NHWC convolution is
    // enabled, while still validating the generic convolution fallback.
    {
      string label("5x5 convolution coopmat NHWC boundary");
      const int largeBatchSize = 1;
      const int largeInChannels = 64;
      const int largeOutChannels = 64;
      const int largeNnXLen = 8;
      const int largeNnYLen = 8;
      constexpr int convSize = 5;
      constexpr int convRadius = convSize / 2;
      vector<float> largeInput((size_t)largeBatchSize * largeInChannels * largeNnXLen * largeNnYLen);
      vector<float> largeWeights((size_t)largeOutChannels * largeInChannels * convSize * convSize);
      for(size_t i = 0; i < largeInput.size(); i++)
        largeInput[i] = (float)((int)(i % 29) - 14) * 0.0234375f;
      for(size_t i = 0; i < largeWeights.size(); i++)
        largeWeights[i] = (float)((int)(i % 31) - 15) * 0.01171875f;
      vector<float> largeExpected((size_t)largeBatchSize * largeOutChannels * largeNnXLen * largeNnYLen, 0.0f);
      for(int oc = 0; oc < largeOutChannels; oc++) {
        for(int y = 0; y < largeNnYLen; y++) {
          for(int x = 0; x < largeNnXLen; x++) {
            float sum = 0.0f;
            for(int ic = 0; ic < largeInChannels; ic++) {
              for(int fy = 0; fy < convSize; fy++) {
                const int iy = y + fy - convRadius;
                if(iy < 0 || iy >= largeNnYLen)
                  continue;
                for(int fx = 0; fx < convSize; fx++) {
                  const int ix = x + fx - convRadius;
                  if(ix < 0 || ix >= largeNnXLen)
                    continue;
                  sum += largeInput[(size_t)ic * largeNnXLen * largeNnYLen + iy * largeNnXLen + ix] *
                         largeWeights[((size_t)(oc * largeInChannels + ic) * convSize + fy) * convSize + fx];
                }
              }
            }
            largeExpected[(size_t)oc * largeNnXLen * largeNnYLen + y * largeNnXLen + x] = sum;
          }
        }
      }
      ConvLayerDesc largeDesc;
      largeDesc.convYSize = convSize;
      largeDesc.convXSize = convSize;
      largeDesc.inChannels = largeInChannels;
      largeDesc.outChannels = largeOutChannels;
      largeDesc.dilationY = 1;
      largeDesc.dilationX = 1;
      largeDesc.weights = largeWeights;
      testConfigurations(label, largeBatchSize, largeNnXLen, largeNnYLen, largeDesc, largeInput, largeExpected);
    }

  }


}


static void testBatchNormLayer(int64_t& numTestsRun) {

  auto testConfigurations = [&](
    const string& label,
    int batchSize, int nnXLen, int nnYLen,
    const BatchNormLayerDesc& desc, const vector<float>& input, const vector<float>& mask, const vector<float>& expected
  ) {
    for(int useNHWC = 0; useNHWC <= 1; useNHWC++) {
      for(int useFP16 = 0; useFP16 <= 1; useFP16++) {
        vector<float> inputThisLoop = useNHWC ? NCHWtoNHWC(input,batchSize,desc.numChannels,nnYLen,nnXLen) : input;
        vector<float> maskThisLoop = mask;
        vector<float> expectedThisLoop = useNHWC ? NCHWtoNHWC(expected,batchSize,desc.numChannels,nnYLen,nnXLen) : expected;

        vector<float> outputThisLoop;
        bool supported = NeuralNet::testEvaluateBatchNorm(
          &desc,batchSize,nnXLen,nnYLen,useFP16,useNHWC,inputThisLoop,maskThisLoop,outputThisLoop
        );

        if(supported) {
          numTestsRun += 1;
          string subLabel = label + Global::strprintf(" useNHWC %d useFP16 %d", useNHWC, useFP16);
          if(useNHWC)
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,nnYLen,nnXLen,desc.numChannels,useFP16);
          else
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,desc.numChannels,nnYLen,nnXLen,useFP16);
        }
      }
    }
  };

  {
    int batchSize = 2;
    int numChannels = 2;
    int nnYLen = 2;
    int nnXLen = 5;

    //NCHW
    vector<float> input({
        5,5,4,4,9,
        1,1,8,8,9,

        0,1,2,3,4,
        8,7,6,5,4,

        3,0,4,0,5,
        0,5,0,6,0,

        1,0,0,2,1,
        0,2,2,0,2,
    });

    {
      string label("Batch norm");

      BatchNormLayerDesc desc;
      desc.numChannels = numChannels;
      desc.epsilon = 0.1f;
      desc.hasScale = true;
      desc.hasBias = true;
      desc.mean = vector<float>({0.0f,2.0f});
      desc.variance = vector<float>({3.9f,0.15f});
      desc.scale = vector<float>({0.1f,1.0f});
      desc.bias = vector<float>({10.0f,0.0f});
      desc.computeMerged();

      vector<float> mask({
        1,1,1,1,1,
        1,1,1,1,1,

        1,1,1,1,1,
        1,1,1,1,1,
      });

      //NCHW
      vector<float> expected({
          10.25f, 10.25f, 10.2f, 10.2f, 10.45f,
          10.05f, 10.05f, 10.4f, 10.4f, 10.45f,

          -4.0f, -2.0f, 0.0f, 2.0f, 4.0f,
          12.0f, 10.0f, 8.0f, 6.0f, 4.0f,

          10.15f, 10.00f, 10.20f, 10.00f, 10.25f,
          10.00f, 10.25f, 10.00f, 10.30f, 10.00f,

          -2.0f, -4.0f, -4.0f, 0.0f, -2.0f,
          -4.0f, 0.0f, 0.0f, -4.0f, 0.0f,
      });
      testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,mask,expected);
    }

    {
      string label("Batch norm with mask");

      BatchNormLayerDesc desc;
      desc.numChannels = numChannels;
      desc.epsilon = 0.1f;
      desc.hasScale = false;
      desc.hasBias = true;
      desc.mean = vector<float>({0.0f,2.0f});
      desc.variance = vector<float>({3.9f,0.15f});
      desc.scale = vector<float>({1.0f,1.0f});
      desc.bias = vector<float>({10.0f,0.0f});
      desc.computeMerged();

      vector<float> mask({
        1,1,1,0,0,
        1,1,1,0,0,

        1,1,1,1,1,
        0,0,0,0,0,
      });

      //NCHW
      vector<float> expected({
          12.5, 12.5, 12, 0, 0,
          10.5, 10.5, 14, 0, 0,

          -4, -2, 0, 0, 0,
          12, 10, 8, 0, 0,

          11.5, 10, 12, 10, 12.5,
          0, 0, 0, 0, 0,

          -2, -4, -4, 0, -2,
          0, 0, 0, 0, 0,
      });

      testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,mask,expected);
    }

  }

  // Exercise the direct Vulkan NHWC test path with both the vec4 channel
  // specialization and an explicit non-multiple-of-four scalar tail.
  auto testGeneratedChannels = [&](int numChannels, bool masked, const string& label) {
    const int batchSize = 2;
    const int nnYLen = 3;
    const int nnXLen = 5;
    const int nnXYLen = nnXLen * nnYLen;

    BatchNormLayerDesc desc;
    desc.numChannels = numChannels;
    desc.epsilon = 0.125f;
    desc.hasScale = true;
    desc.hasBias = true;
    desc.mean.resize(numChannels);
    desc.variance.resize(numChannels);
    desc.scale.resize(numChannels);
    desc.bias.resize(numChannels);
    for(int c = 0; c < numChannels; c++) {
      desc.mean[c] = (float)(c - 2) * 0.125f;
      desc.variance[c] = 0.75f + (float)(c % 3) * 0.25f;
      desc.scale[c] = 0.5f + (float)(c % 5) * 0.125f;
      desc.bias[c] = (float)(3 - c) * 0.0625f;
    }
    desc.computeMerged();

    vector<float> input((size_t)batchSize * numChannels * nnXYLen);
    vector<float> mask((size_t)batchSize * nnXYLen, 1.0f);
    vector<float> expected(input.size());
    for(size_t i = 0; i < input.size(); i++)
      input[i] = (float)((int)(i % 29) - 14) * 0.09375f;
    if(masked) {
      for(size_t i = 0; i < mask.size(); i++)
        mask[i] = (i % 7 == 2 || i % 11 == 5) ? 0.0f : 1.0f;
    }
    for(int n = 0; n < batchSize; n++) {
      for(int c = 0; c < numChannels; c++) {
        for(int xy = 0; xy < nnXYLen; xy++) {
          size_t idx = ((size_t)n * numChannels + c) * nnXYLen + xy;
          expected[idx] = (input[idx] * desc.mergedScale[c] + desc.mergedBias[c]) * mask[n * nnXYLen + xy];
        }
      }
    }
    testConfigurations(label, batchSize, nnXLen, nnYLen, desc, input, mask, expected);
  };

  testGeneratedChannels(8, false, "Batch norm vec4 channels");
  testGeneratedChannels(8, true, "Batch norm vec4 channels with mask");
  testGeneratedChannels(5, true, "Batch norm scalar channel tail");
}


static void testResidualBlock(int64_t& numTestsRun) {

  auto testConfigurations = [&](
    const string& label,
    int batchSize, int nnXLen, int nnYLen,
    const ResidualBlockDesc& desc, const vector<float>& input, const vector<float>& mask, const vector<float>& expected
  ) {
    for(int useNHWC = 0; useNHWC <= 1; useNHWC++) {
      for(int useFP16 = 0; useFP16 <= 1; useFP16++) {
        vector<float> inputThisLoop = useNHWC ? NCHWtoNHWC(input,batchSize,desc.preBN.numChannels,nnYLen,nnXLen) : input;
        vector<float> maskThisLoop = mask;
        vector<float> expectedThisLoop = useNHWC ? NCHWtoNHWC(expected,batchSize,desc.preBN.numChannels,nnYLen,nnXLen) : expected;

        vector<float> outputThisLoop;
        bool supported = NeuralNet::testEvaluateResidualBlock(
          &desc,batchSize,nnXLen,nnYLen,useFP16,useNHWC,inputThisLoop,maskThisLoop,outputThisLoop
        );

        if(supported) {
          numTestsRun += 1;
          string subLabel = label + Global::strprintf(" useNHWC %d useFP16 %d", useNHWC, useFP16);
          if(useNHWC)
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,nnYLen,nnXLen,desc.preBN.numChannels,useFP16);
          else
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,desc.preBN.numChannels,nnYLen,nnXLen,useFP16);
        }
      }
    }
  };

  {
    string label("Basic residual block");

    int batchSize = 2;
    int trunkChannels = 1;
    int midChannels = 2;
    int nnYLen = 3;
    int nnXLen = 4;

    //NCHW
    vector<float> input({
      1,0,0,0,
      0,2,2,0,
      0,0,0,1,

      0,0,0,0,
      0,3,-5,0,
      1,1,1,1,
    });

    //Also, mask out some values
    vector<float> mask({
      1,1,0,1,
      1,1,1,1,
      1,1,0,1,

      1,1,1,1,
      1,1,1,0,
      1,1,1,1,
    });

    ResidualBlockDesc desc;

    //Doubles all values
    desc.preBN.name = "preBN";
    desc.preBN.numChannels = trunkChannels;
    desc.preBN.epsilon = 0.1f;
    desc.preBN.hasScale = true;
    desc.preBN.hasBias = true;
    desc.preBN.mean = vector<float>({0});
    desc.preBN.variance = vector<float>({0.9f});
    desc.preBN.scale = vector<float>({2});
    desc.preBN.bias = vector<float>({0});
    desc.preBN.computeMerged();

    //ReLU gets applied, smooshing negatives
    //2,0,0,3,
    //0,4,4,0,
    //0,0,0,2,

    //0,0,0,0,
    //0,6,0,0,
    //2,2,2,2,

    //Split into two channels, shifting up and shifting down.
    desc.regularConv.name = "regularConv";
    desc.regularConv.convYSize = 3;
    desc.regularConv.convXSize = 3;
    desc.regularConv.inChannels = trunkChannels;
    desc.regularConv.outChannels = midChannels;
    desc.regularConv.dilationY = 1;
    desc.regularConv.dilationX = 1;
    desc.regularConv.weights = vector<float>({
        0,1,0,
        0,0,0,
        0,0,0,

        0,0,0,
        0,0,0,
        0,1,0,
    });
    //0,0,0,0,
    //2,0,0,3,
    //0,4,0,0,

    //0,4,0,0,
    //0,0,0,2,
    //0,0,0,0,

    //0,0,0,0,
    //0,0,0,0,
    //0,6,0,0,

    //0,6,0,0,
    //2,2,2,0,
    //0,0,0,0,

    //Subtract 3 from all values in the 0th channel
    desc.midBN.name = "midBN";
    desc.midBN.numChannels = midChannels;
    desc.midBN.epsilon = 0.1f;
    desc.midBN.hasScale = false;
    desc.midBN.hasBias = false;
    desc.midBN.mean = vector<float>({3,0});
    desc.midBN.variance = vector<float>({0.9f,0.9f});
    desc.midBN.scale = vector<float>({1,1});
    desc.midBN.bias = vector<float>({0,0});
    desc.midBN.computeMerged();

    //ReLU gets applied, smooshing negatives
    //0,0,0,0,
    //0,0,0,0,
    //0,1,0,0,

    //0,4,0,0,
    //0,0,0,2,
    //0,0,0,0,

    //0,0,0,0,
    //0,0,0,0,
    //0,3,0,0,

    //0,6,0,0,
    //2,2,2,0,
    //0,0,0,0,


    // Sum pointwise. Use a center-only 3x3 kernel so Vulkan's Winograd final-conv path
    //  exercises fused residual addition while preserving the same expected values.
    desc.finalConv.name = "finalConv";
    desc.finalConv.convYSize = 3;
    desc.finalConv.convXSize = 3;
    desc.finalConv.inChannels = midChannels;
    desc.finalConv.outChannels = trunkChannels;
    desc.finalConv.dilationY = 1;
    desc.finalConv.dilationX = 1;
    desc.finalConv.weights = vector<float>({
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,

        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    });

    //0,4,0,0,
    //0,0,0,2,
    //0,1,0,0,

    //0,6,0,0,
    //2,2,2,0,
    //0,3,0,0,

    //Then add to the original which was:

    //1,0,0,0,
    //0,2,2,0,
    //0,0,0,1,

    //0,0,0,0,
    //0,3,-5,0,
    //1,1,1,1,

    //Result:

    //1,4,0,0,
    //0,2,2,2,
    //0,1,0,1,

    //0,6,0,0,
    //2,5,-3,0,
    //1,4,1,1,


    //NCHW
    vector<float> expected({
        1, 4, 0, 0,
        0, 2, 2, 2,
        0, 1, 0, 1,

        0, 6, 0, 0,
        2, 5, -3, 0,
        1, 4, 1, 1,
    });

    testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,mask,expected);

    label = "Basic residual block 1x1 final";
    desc.finalConv.convYSize = 1;
    desc.finalConv.convXSize = 1;
    desc.finalConv.weights = vector<float>({1, 1});

    testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,mask,expected);
  }

}

static void testGlobalPoolingResidualBlock(int64_t& numTestsRun) {

  auto testConfigurations = [&](
    const string& label,
    int batchSize, int nnXLen, int nnYLen,
    const GlobalPoolingResidualBlockDesc& desc, const vector<float>& input, const vector<float>& mask, const vector<float>& expected
  ) {
    for(int useNHWC = 0; useNHWC <= 1; useNHWC++) {
      for(int useFP16 = 0; useFP16 <= 1; useFP16++) {
        vector<float> inputThisLoop = useNHWC ? NCHWtoNHWC(input,batchSize,desc.preBN.numChannels,nnYLen,nnXLen) : input;
        vector<float> maskThisLoop = mask;
        vector<float> expectedThisLoop = useNHWC ? NCHWtoNHWC(expected,batchSize,desc.preBN.numChannels,nnYLen,nnXLen) : expected;

        vector<float> outputThisLoop;
        bool supported = NeuralNet::testEvaluateGlobalPoolingResidualBlock(
          &desc,batchSize,nnXLen,nnYLen,useFP16,useNHWC,inputThisLoop,maskThisLoop,outputThisLoop
        );

        if(supported) {
          numTestsRun += 1;
          string subLabel = label + Global::strprintf(" useNHWC %d useFP16 %d", useNHWC, useFP16);
          if(useNHWC)
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,nnYLen,nnXLen,desc.preBN.numChannels,useFP16);
          else
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,desc.preBN.numChannels,nnYLen,nnXLen,useFP16);
        }
      }
    }
  };

  {
    string label("Global pooling residual block");

    int batchSize = 2;
    int trunkChannels = 1;
    int regularChannels = 1;
    int gpoolChannels = 2;
    int nnYLen = 3;
    int nnXLen = 4;

    //NCHW
    vector<float> input({
      1,2,0,0,
      0,3,4,0,
      0,0,5,0,

      0,0,0,0,
      0,5,-3,0,
      0,-1,1,1,
    });

    vector<float> mask({
      1,1,1,0,
      1,1,1,0,
      1,1,1,0,

      0,0,0,0,
      0,1,1,1,
      0,1,1,1,
    });

    GlobalPoolingResidualBlockDesc desc;

    //Identity map
    desc.preBN.name = "preBN";
    desc.preBN.numChannels = trunkChannels;
    desc.preBN.epsilon = 0.1f;
    desc.preBN.hasScale = true;
    desc.preBN.hasBias = true;
    desc.preBN.mean = vector<float>({0});
    desc.preBN.variance = vector<float>({0.9f});
    desc.preBN.scale = vector<float>({1});
    desc.preBN.bias = vector<float>({0});
    desc.preBN.computeMerged();

    //ReLU gets applied, smooshing negatives
    //1,2,0,0,
    //0,3,4,0,
    //0,0,5,0,

    //0,0,0,0,
    //0,5,0,0,
    //0,0,1,1,

    //Double the value
    desc.regularConv.name = "regularConv";
    desc.regularConv.convYSize = 1;
    desc.regularConv.convXSize = 1;
    desc.regularConv.inChannels = trunkChannels;
    desc.regularConv.outChannels = regularChannels;
    desc.regularConv.dilationY = 1;
    desc.regularConv.dilationX = 1;
    desc.regularConv.weights = vector<float>({
        2
    });
    //2,4,0,0,
    //0,6,8,0,
    //0,0,10,0,

    //0,0,0,0,
    //0,10,0,0,
    //0,0,2,2,

    //For gpooling, split into two channels, shifting left and right
    desc.gpoolConv.name = "gpoolConv";
    desc.gpoolConv.convYSize = 3;
    desc.gpoolConv.convXSize = 3;
    desc.gpoolConv.inChannels = trunkChannels;
    desc.gpoolConv.outChannels = gpoolChannels;
    desc.gpoolConv.dilationY = 1;
    desc.gpoolConv.dilationX = 1;
    desc.gpoolConv.weights = vector<float>({
        0,0,0,
        0,0,1,
        0,0,0,

        0,0,0,
        1,0,0,
        0,0,0,
    });
    //2,0,0,0,
    //3,4,0,0,
    //0,5,0,0,

    //0,1,2,0,
    //0,0,3,0,
    //0,0,0,0,

    //0,0,0,0,
    //0,0,0,0,
    //0,1,1,0,

    //0,0,0,0,
    //0,0,5,0,
    //0,0,0,1,

    //Subtract 2 from all values in the 1th channel
    desc.gpoolBN.name = "gpoolBN";
    desc.gpoolBN.numChannels = gpoolChannels;
    desc.gpoolBN.epsilon = 0.1f;
    desc.gpoolBN.hasScale = false;
    desc.gpoolBN.hasBias = false;
    desc.gpoolBN.mean = vector<float>({0,0});
    desc.gpoolBN.variance = vector<float>({0.9f,0.9f});
    desc.gpoolBN.scale = vector<float>({1,1});
    desc.gpoolBN.bias = vector<float>({0,-2});
    desc.gpoolBN.computeMerged();

    //And apply RELU

    //2,0,0,0,
    //3,4,0,0,
    //0,5,0,0,

    //0,0,0,0,
    //0,0,1,0,
    //0,0,0,0,

    //0,0,0,0,
    //0,0,0,0,
    //0,1,1,0,

    //0,0,0,0,
    //0,0,3,0,
    //0,0,0,0,

    //Pooling - mean, mean * (sqrt(masksum) - 14) * 0.1, max

    //14.0/9.0, 14.0/9.0*(-11)*0.1, 5
    //1.0/9.0, 1.0/9.0*(-11)*0.1, 1

    //2.0/6.0, 2.0/6.0*(sqrt(6)-14)*0.1, 1
    //3.0/6.0, 3.0/6.0*(sqrt(6)-14)*0.1, 3

    //Recombine values
    desc.gpoolToBiasMul.inChannels = 6;
    desc.gpoolToBiasMul.outChannels = regularChannels;
    desc.gpoolToBiasMul.weights = vector<float>({36,36, 18,18, 1,1});

    //56 + 28*(-11)*0.1 + 5 +
    //4 + 2*(-11)*0.1 + 1

    //12 + 6*(sqrt(6)-14)*0.1 + 1 +
    //18 + 9*(sqrt(6)-14)*0.1 + 3

    //Identity map
    desc.midBN.name = "midBN";
    desc.midBN.numChannels = regularChannels;
    desc.midBN.epsilon = 0.1f;
    desc.midBN.hasScale = false;
    desc.midBN.hasBias = false;
    desc.midBN.mean = vector<float>({0});
    desc.midBN.variance = vector<float>({0.9f});
    desc.midBN.scale = vector<float>({1});
    desc.midBN.bias = vector<float>({0});
    desc.midBN.computeMerged();

    //Relu gets applied, should hit nothing in this case

    // Identity map. Use a center-only 3x3 kernel so Vulkan's Winograd final-conv path
    //  exercises fused residual addition while preserving the same expected values.
    desc.finalConv.name = "finalConv";
    desc.finalConv.convYSize = 3;
    desc.finalConv.convXSize = 3;
    desc.finalConv.inChannels = regularChannels;
    desc.finalConv.outChannels = trunkChannels;
    desc.finalConv.dilationY = 1;
    desc.finalConv.dilationX = 1;
    desc.finalConv.weights = vector<float>({
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    });

    vector<float> expected({
      3,6,0,0,
      0,9,12,0,
      0,0,15,0,

      0,0,0,0,
      0,15,-3,0,
      0,-1,3,3,
    });

    for(int i = 0; i<12; i++) {
      expected[i] += (float)(
        56 + 28*(-11)*0.1 + 5 +
        4 + 2*(-11)*0.1 + 1
      );
      expected[i] *= mask[i];
    }
    for(int i = 12; i<24; i++) {
      expected[i] += (float)(
        12 + 6*(sqrt(6)-14)*0.1 + 1 +
        18 + 9*(sqrt(6)-14)*0.1 + 3
      );
      expected[i] *= mask[i];
    }

    testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,mask,expected);

    label = "Global pooling residual block 1x1 final";
    desc.finalConv.convYSize = 1;
    desc.finalConv.convXSize = 1;
    desc.finalConv.weights = vector<float>({1});

    testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,mask,expected);
  }

}


void Tests::runNNLayerTests() {
  NeuralNet::globalInitialize();
  int64_t numTestsRun = 0;
  testConvLayer(numTestsRun);
  testBatchNormLayer(numTestsRun);
  testResidualBlock(numTestsRun);
  testGlobalPoolingResidualBlock(numTestsRun);
  NeuralNet::globalCleanup();
  cout << "Tested " << numTestsRun << " configurations" << endl;
  cout << "Done" << endl;
}


void Tests::runNNSymmetryTests() {
  auto testConfigurations = [&](
    const string& label,
    int batchSize, int numChannels, int nnXLen, int nnYLen,
    const vector<float>& input
  ) {
    for(int useNHWC = 0; useNHWC <= 1; useNHWC++) {
      for(int symmetry = 0; symmetry < 8; symmetry++) {
        vector<float> inputThisLoop = useNHWC ? NCHWtoNHWC(input,batchSize,numChannels,nnYLen,nnXLen) : input;
        vector<float> outputThisLoop(inputThisLoop.size());
        SymmetryHelpers::copyInputsWithSymmetry(
          inputThisLoop.data(),outputThisLoop.data(),batchSize,nnXLen,nnYLen,numChannels,useNHWC,symmetry
        );
        cout << label << " useNHWC " << useNHWC << " " << symmetry << endl;
        for(int i = 0; i<outputThisLoop.size(); i++)
          cout << outputThisLoop[i] << " ";
        cout << endl;
      }
    }
    for(int symmetry = 0; symmetry < 8; symmetry++) {
      vector<float> inputThisLoop = input;
      vector<float> outputThisLoop(inputThisLoop.size());
      SymmetryHelpers::copyOutputsWithSymmetry(
        inputThisLoop.data(),outputThisLoop.data(),batchSize*numChannels,nnXLen,nnYLen,symmetry
      );
      cout << label << " OUTPUT " << endl;
      for(int i = 0; i<outputThisLoop.size(); i++)
        cout << outputThisLoop[i] << " ";
      cout << endl;
    }
  };

  {
    {
      //NCHW
      vector<float> input({
        0,1,2,
        3,4,5,
        6,7,8,

        3,0,4,
        0,5,0,
        0,6,0,

        1,0,0,
        1,1,1,
        1,0,1,
      });
      testConfigurations("Symmetry 3-1-3-3",3,1,3,3,input);
      testConfigurations("Symmetry 1-3-3-3",1,3,3,3,input);
    }

    {
      //NCHW
      vector<float> input({
        0,1,2,3,
        4,5,6,7,
        8,9,10,11,

        12,13,14,15,
        16,17,18,19,
        20,21,22,23
      });

      testConfigurations("Symmetry 2-1-3-4",2,1,3,4,input);
      testConfigurations("Symmetry 2-3-2-2",2,3,2,2,input);
    }

  }

}
