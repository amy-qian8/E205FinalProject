// STM32F401RE_RCC.c
// Source code for RCC functions

#include "STM32F401RE_RCC.h"

void configurePLL() {
    // Set clock to 84 MHz
    // Output freq = (src_clk) * (N/M) / P
    // (8 MHz) * (336/16) / 4 = 42 MHz
    // M:16, N:336, P:4, Q:7
    // Use HSE as PLLSRC

    RCC->CR.PLLON = 0; // Turn off PLL
    while (RCC->CR.PLLRDY != 0); // Wait till PLL is unlocked (e.g., off)

    // Load configuration
    RCC->PLLCFGR.PLLSRC = PLLSRC_HSE;
    RCC->PLLCFGR.PLLM = 8;
    RCC->PLLCFGR.PLLN = 336;
    RCC->PLLCFGR.PLLP = 0b01;
    RCC->PLLCFGR.PLLQ = 4;

    // Enable PLL and wait until it's locked
    RCC->CR.PLLON = 1;
    while(RCC->CR.PLLRDY == 0);
}

void configureClock(){
        // Configure prescalers
    RCC->CFGR.PPRE2 = 0b000;
    RCC->CFGR.PPRE1 = 0b100;

    // Turn on and bypass for HSE from ST-LINK
    RCC->CR.HSEBYP = 1;
    RCC->CR.HSEON = 1;
    while(!RCC->CR.HSERDY);
    
    // Configure and turn on PLL
    configurePLL();

    // Select PLL as clock source
    RCC->CFGR.SW = SW_PLL;
    while(RCC->CFGR.SWS != 0b10);
}