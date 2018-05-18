module P_to_IC(in0, in1, in2, in3, in4, in5, in6, in7, in8, clk, rst_b, ret_sig,
	C_matrix_0, C_matrix_1, C_matrix_2, C_matrix_3, C_matrix_4, C_matrix_5, C_matrix_6, C_matrix_7, C_matrix_8,
	C_matrix_9, C_matrix_10, C_matrix_11, C_matrix_12, C_matrix_13, C_matrix_14, C_matrix_15, C_matrix_16, C_matrix_17,
	C_matrix_18, C_matrix_19, C_matrix_20, C_matrix_21, C_matrix_22, C_matrix_23, C_matrix_24, C_matrix_25, C_matrix_26,
	I_matrix_0, I_matrix_1, I_matrix_2, I_matrix_3, I_matrix_4, I_matrix_5, I_matrix_6, I_matrix_7, I_matrix_8,
	I_matrix_9, I_matrix_10, I_matrix_11, I_matrix_12, I_matrix_13, I_matrix_14, I_matrix_15, I_matrix_16, I_matrix_17,
	I_matrix_18, I_matrix_19, I_matrix_20, I_matrix_21, I_matrix_22, I_matrix_23, I_matrix_24, I_matrix_25, I_matrix_26 );
	
	parameter DATA_WIDTH = 8;
	parameter s =  3;
	parameter kh = 3;
	parameter kw = 3;
	parameter k = 10;
	
	input [DATA_WIDTH-1:0] in0, in1, in2, in3, in4, in5, in6, in7, in8;
	input clk, rst_b;
	output ret_sig;
	
	output [DATA_WIDTH-1:0]	C_matrix_0, C_matrix_1, C_matrix_2, C_matrix_3, C_matrix_4, C_matrix_5, C_matrix_6, C_matrix_7, C_matrix_8,
	C_matrix_9, C_matrix_10, C_matrix_11, C_matrix_12, C_matrix_13, C_matrix_14, C_matrix_15, C_matrix_16, C_matrix_17,
	C_matrix_18, C_matrix_19, C_matrix_20, C_matrix_21, C_matrix_22, C_matrix_23, C_matrix_24, C_matrix_25, C_matrix_26;
	
	output [DATA_WIDTH-1:0]	I_matrix_0, I_matrix_1, I_matrix_2, I_matrix_3, I_matrix_4, I_matrix_5, I_matrix_6, I_matrix_7, I_matrix_8,
	I_matrix_9, I_matrix_10, I_matrix_11, I_matrix_12, I_matrix_13, I_matrix_14, I_matrix_15, I_matrix_16, I_matrix_17,
	I_matrix_18, I_matrix_19, I_matrix_20, I_matrix_21, I_matrix_22, I_matrix_23, I_matrix_24, I_matrix_25, I_matrix_26;

	reg ret_sig;
	reg [DATA_WIDTH-1:0] C_matrix [0:26];
	reg [DATA_WIDTH-1:0] I_matrix [0:26];
	
	reg [DATA_WIDTH-1:0] k_index; //internal for k_index counter
	integer i, flag0, flag1, flag2, flag3, flag4, flag5, flag6, flag7, flag8;
	
	assign C_matrix_0 = C_matrix[0]; assign C_matrix_1 = C_matrix[1]; assign C_matrix_2 =C_matrix[2];
	assign C_matrix_3 = C_matrix[3]; assign C_matrix_4 = C_matrix[4]; assign C_matrix_5 =C_matrix[5];
	assign C_matrix_6 = C_matrix[6]; assign C_matrix_7 = C_matrix[7]; assign C_matrix_8 =C_matrix[8];
	assign C_matrix_9 = C_matrix[9]; assign C_matrix_10 = C_matrix[10]; assign C_matrix_11 =C_matrix[11];
	assign C_matrix_12 = C_matrix[12]; assign C_matrix_13 = C_matrix[13]; assign C_matrix_14 =C_matrix[14];
	assign C_matrix_15 = C_matrix[15]; assign C_matrix_16 = C_matrix[16]; assign C_matrix_17 =C_matrix[17];
	assign C_matrix_18 = C_matrix[18]; assign C_matrix_19 = C_matrix[19]; assign C_matrix_20 =C_matrix[20];
	assign C_matrix_21 = C_matrix[21]; assign C_matrix_22 = C_matrix[22]; assign C_matrix_23 =C_matrix[23];
	assign C_matrix_24 = C_matrix[24]; assign C_matrix_25 = C_matrix[25]; assign C_matrix_26 =C_matrix[26];
	
	assign I_matrix_0 = I_matrix[0]; assign I_matrix_1 = I_matrix[1]; assign I_matrix_2 = I_matrix[2];
	assign I_matrix_3 = I_matrix[3]; assign I_matrix_4 = I_matrix[4]; assign I_matrix_5 =I_matrix[5];
	assign I_matrix_6 = I_matrix[6]; assign I_matrix_7 = I_matrix[7]; assign I_matrix_8 =I_matrix[8];
	assign I_matrix_9 = I_matrix[9]; assign I_matrix_10 = I_matrix[10]; assign I_matrix_11 = I_matrix[11];
	assign I_matrix_12 = I_matrix[12]; assign I_matrix_13 = I_matrix[13]; assign I_matrix_14 =I_matrix[14];
	assign I_matrix_15 = I_matrix[15]; assign I_matrix_16 = I_matrix[16]; assign I_matrix_17 = I_matrix[17];
	assign I_matrix_18 = I_matrix[18]; assign I_matrix_19 = I_matrix[19]; assign I_matrix_20 =I_matrix[20];
	assign I_matrix_21 = I_matrix[21]; assign I_matrix_22 = I_matrix[22]; assign I_matrix_23 =I_matrix[23];
	assign I_matrix_24 = I_matrix[24]; assign I_matrix_25 = I_matrix[25]; assign I_matrix_26 =I_matrix[26];
	
	always@(clk) //calculate in every time slot(1ns)...level trigger
	begin
		if(rst_b)  //reset stage
		begin			
			for(i = s*kh*kw-1; i >= 0; i=i-1)
			begin
				C_matrix[i] = 0;
				I_matrix[i] = 0;	
			end
			
			k_index = 1;
			ret_sig = 0;
		end
		else if(k_index > k)
		begin    //output stage
		    ret_sig = 1;
		end
		else      //normal calculate stage
		begin
			if(in0 != 0) begin
			    flag0 = 0;
			    for(i = 0; i < s; i=i+1) begin
				    if(I_matrix[i*kw*kh+0]==0 && flag0==0) begin
					    I_matrix[i*kw*kh+0] = k_index;
						C_matrix[i*kw*kh+0] = in0;
						flag0 = 1;
					end
				end
			end
			if(in1 != 0) begin
			    flag1 = 0;  //filling any slot makes flag = 1
			    for(i = 0; i < s; i=i+1) begin
				    if(I_matrix[i*kw*kh+1]==0 && flag1==0) begin
					    I_matrix[i*kw*kh+1] = k_index; //fill slot
						C_matrix[i*kw*kh+1] = in1;
						flag1 = 1;
					end
				end
			end
			if(in2 != 0) begin
			    flag2 = 0;
			    for(i = 0; i < s; i=i+1) begin
				    if(I_matrix[i*kw*kh+2]==0 && flag2==0) begin
					    I_matrix[i*kw*kh+2] = k_index;
						C_matrix[i*kw*kh+2] = in2;
						flag2 = 1;
					end
				end
			end
			if(in3 != 0) begin
			    flag3 = 0;
			    for(i = 0; i < s; i=i+1) begin
				    if(I_matrix[i*kw*kh+3]==0 && flag3==0) begin
					    I_matrix[i*kw*kh+3] = k_index;
						C_matrix[i*kw*kh+3] = in3;
						flag3 = 1;
					end
				end
			end
			if(in4 != 0) begin
			    flag4 = 0;
			    for(i = 0; i < s; i=i+1) begin
				    if(I_matrix[i*kw*kh+4]==0 && flag4==0) begin
					    I_matrix[i*kw*kh+4] = k_index;
						C_matrix[i*kw*kh+4] = in4;
						flag4 = 1;
					end
				end
			end
			if(in5 != 0) begin
			    flag5 = 0;
			    for(i = 0; i < s; i=i+1) begin
				    if(I_matrix[i*kw*kh+5]==0 && flag5==0) begin
					    I_matrix[i*kw*kh+5] = k_index;
						C_matrix[i*kw*kh+5] = in5;
						flag5 = 1;
					end
				end
			end
			if(in6 != 0) begin
			    flag6 = 0;
			    for(i = 0; i < s; i=i+1) begin
				    if(I_matrix[i*kw*kh+6]==0 && flag6==0) begin
					    I_matrix[i*kw*kh+6] = k_index;
						C_matrix[i*kw*kh+6] = in6;
						flag6 = 1;
					end
				end
			end
			if(in7 != 0) begin
			    flag7 = 0;
			    for(i = 0; i < s; i=i+1) begin
				    if(I_matrix[i*kw*kh+7]==0 && flag7==0) begin
					    I_matrix[i*kw*kh+7] = k_index;
						C_matrix[i*kw*kh+7] = in7;
						flag7 = 1;
					end
				end
			end
			if(in8 != 0) begin
			    flag8 = 0;
			    for(i = 0; i < s; i=i+1) begin
				    if(I_matrix[i*kw*kh+8]==0 && flag8==0) begin
					    I_matrix[i*kw*kh+8] = k_index;
						C_matrix[i*kw*kh+8] = in8;
						flag8 = 1;
					end
				end
			end
			
			if(k_index == k) begin
				ret_sig = 1;
			end
			k_index = k_index + 1;
		end
	end  //end clk 
endmodule
