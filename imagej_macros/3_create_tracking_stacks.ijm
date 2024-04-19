tot_pos=22;
tube_width=730; // px
delta_t=3; // min
min_ratio=0.3;
max_ratio=1.6;
// Assumes labels are in \Labels, bright-field images in \Processed_Data, and ratio stacks in \Ratio
data_dir="Z:\\Users\\Franco\\Experiments\\10-28-22"; 
save_dir="Z:\\Users\\Franco\\Experiments\\10-28-22\\Tracking_Stacks";

setBatchMode(true);
um_to_px=2000/tube_width; // 2000 um is the width of the tube
setOption("ScaleConversions", true);
File.makeDirectory(save_dir);
for (pos=0; pos<tot_pos; pos=pos+1){
	// Labels
    File.openSequence(data_dir+"/Labels/Pos"+pos+"/", "virtual");
    rename("Pos"+pos+"_Labels");
    frames=nSlices();
    Stack.setXUnit("um");
    Stack.setYUnit("um");
    run("Properties...", "channels=1 slices=1 frames="+frames+" unit=um pixel_width="+um_to_px+" pixel_height="+um_to_px+" voxel_depth=1.0000000 frame="+delta_t+" origin=0,0");
    // Bright-field
    File.openSequence(data_dir+"/Processed_Data/Pos"+pos+"/", "virtual filter=BF");
    rename("Pos"+pos+"_BF");
    Stack.setXUnit("um");
    Stack.setYUnit("um");
    run("Properties...", "channels=1 slices=1 frames="+frames+" unit=um pixel_width="+um_to_px+" pixel_height="+um_to_px+" voxel_depth=1.0000000 frame="+delta_t+" origin=0,0");
    run("16-bit");
    // Ratio
    open(data_dir+"\\Ratio\\Pos"+pos+"_Ratio.tif");
    rename("Pos"+pos+"_Ratio");
    Stack.setXUnit("um");
    Stack.setYUnit("um");
    run("Properties...", "channels=1 slices=1 frames="+frames+" unit=um pixel_width="+um_to_px+" pixel_height="+um_to_px+" voxel_depth=1.0000000 frame="+delta_t+" origin=0,0");
    setMinAndMax(min_ratio, max_ratio);
    run("16-bit");
    // Combine stacks
    run("Merge Channels...", "c2=Pos"+pos+"_Labels c3=Pos"+pos+"_Ratio c4=Pos"+pos+"_BF create keep");
    // Save
    saveAs("Tiff", save_dir+"\\Pos"+pos+"_Tracking.tif");
	close("*");
}