// Spliced into the importer with #include "lib/shapes.fs".
int rect_area(int w, int h) { return w * h; }
int circle_area_scaled(int r) { int a = 314 * r * r; return a / 100; }
