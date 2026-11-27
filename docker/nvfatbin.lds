/* nvfatbin.lds -- linker-script fragment for CUDA builds.
 *
 * Park the fatbin in its own output section at the very end of the
 * image, so .text/.rodata/.got/.data all stay within 2 GB of each other.
 *
 * Used via:  -Wl,--script=/app/docker/nvfatbin.lds
 */

SECTIONS {
  .nvfatbin : { *(.nv_fatbin) }
} INSERT AFTER .bss;
