CU_APPS=linear_transform aggregate_features \
        compute_attention_coeff softmax \
C_APPS=gat

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -arch=sm_86 -o $@ $<
%: %.cpp
	g++ -O2 -std=c++11 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
