#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <vlc/plugins/vlc_common.h>
#include <vlc/plugins/vlc_plugin.h>
#include <vlc/plugins/vlc_filter.h>
#include <vlc/plugins/vlc_picture.h>
#include <vlc/plugins/vlc_variables.h>

#include "plugin.h"

// Including the vfgs project
#include "../src/vfgs_hw.h"

/* Internationalization */
#define DOMAIN "intergrain"
#define _(str) dgettext(DOMAIN, str)
#define N_(str) (str)

#define MODULE_STRING "intergrain"
#define MY_FILTER_NAME "Test Grain Filter"
#define FILTER_PREFIX "intergrain-"
#define HELP_STR "Add random grain to the image"

static char depth;
void (*ptr_add_grain)(void* Y, void* U, void* V, unsigned y, unsigned width, unsigned height, unsigned stride, unsigned cstride);

const char *const ppsz_filter_options[] = {
    "enabled", "grain-amount", NULL};

static int supportedChroma(vlc_fourcc_t *i_chroma){
    /**
     * Return 1 if the chroma input is into the supported chromas
     * @i_chroma chroma to be found
    */
    const vlc_fourcc_t supported_formats[] = 
        {VLC_CODEC_I420, VLC_CODEC_I420_10B,
         VLC_CODEC_I422, VLC_CODEC_I422_10B,
         VLC_CODEC_I444, VLC_CODEC_I444_10B
        };
    int len = sizeof(supported_formats) / sizeof(supported_formats[0]);
    int i;
    for (i = 0; i < len; i++)
    {
        if (supported_formats[i] == i_chroma)
        {
            return 1;
        }
    }
    return 0;
}

static int Open(vlc_object_t *p_this)
{
    msg_Dbg(p_this, N_("Open function"));
    filter_t *p_filter = (filter_t *)p_this;

    const vlc_chroma_description_t *chroma =
        vlc_fourcc_GetChromaDescription(p_filter->fmt_in.video.i_chroma); //Set information about chroma -> i_planes

    // Showing the video format in integer and their equivalent in char
    const vlc_fourcc_t *i_chroma = p_filter->fmt_in.video.i_chroma;

    // Check that the chroma has a correct format
    // Check if the chroma format is one of the supported formats available for the Intergrain plugin
    if (!chroma || chroma->plane_count < 3 || chroma->pixel_size != 1 || !supportedChroma(i_chroma))
    {
        msg_Err(p_filter, "Unsupported chroma (%4.4s)",
                (char*)&(i_chroma));
        return VLC_EGENERIC;
    }

    msg_Info(p_this, N_("The chroma format is: (%4.4s), it is supported!"), (char*)&i_chroma);

    // Calling the add_grain function corresponding with the chroma format
    if(i_chroma == VLC_CODEC_I420)
    {
        ptr_add_grain = vfgs_add_grain_stripe_420_8bits;
        depth = 1;
    } 
    else if(i_chroma == VLC_CODEC_I420_10B)
    {
        ptr_add_grain = vfgs_add_grain_stripe_420_10bits;
        depth = 2;
    } 
    else if(i_chroma == VLC_CODEC_I422)
    {
        ptr_add_grain = vfgs_add_grain_stripe_422_8bits;
        depth = 1;
    }
    else if(i_chroma == VLC_CODEC_I422_10B)
    {
        ptr_add_grain = vfgs_add_grain_stripe_422_10bits;
        depth = 2;
    } 
    else if(i_chroma == VLC_CODEC_I444)
    {
        ptr_add_grain = vfgs_add_grain_stripe_444_8bits;
        depth = 1;
    } 
    else if(i_chroma == VLC_CODEC_I444_10B)
    {
        ptr_add_grain = vfgs_add_grain_stripe_444_10bits;
        depth = 2;
    }

    msg_Dbg(p_this, N_("The value of the depth is: (%d)"), depth);

    // Memory allocation
    my_filter_sys_t *p_sys = p_filter->p_sys = malloc(sizeof(*p_sys));
    msg_Dbg(p_this, N_("After malloc"));
    if (!p_sys)
        return VLC_ENOMEM;

    // Adding the checkbox to activate the plugin
    // Adding the number selection box
    p_sys->i_grain_amount = var_CreateGetIntegerCommand(p_filter, FILTER_PREFIX "grain-amount");
    p_sys->b_enabled = var_CreateGetBool(p_this, FILTER_PREFIX "enabled");

    // Calling the function to add the grain
    var_AddCallback(p_filter, FILTER_PREFIX "grain-amount", FilterCallback, p_sys);

    p_filter->pf_video_filter = Filter;
    vlc_mutex_init(&p_sys->lock);
    msg_Dbg(p_filter, N_("Plugin InterGrain charged"));

    return VLC_SUCCESS;
}

static void Close(vlc_object_t *p_this)
{
    filter_t *p_filter = (filter_t *)p_this;
    my_filter_sys_t *p_sys = p_filter->p_sys;
    var_DelCallback(p_filter, FILTER_PREFIX "grain-amount",FilterCallback, p_sys);
    vlc_mutex_destroy(&p_sys->lock);

    free(p_sys);
}

static picture_t *Filter(filter_t *p_filter, picture_t *p_pic_in)
{    
    my_filter_sys_t *p_sys = p_filter->p_sys;

    if (!p_sys->b_enabled)
    {
        msg_Dbg(p_filter, N_("In Filter function b_enabled is False"));
        
        // If it doesn't work, only return p_pic_in and comment this line.
        return p_pic_in;
    }

    picture_t *p_pic_out = filter_NewPicture(p_filter);
    if (!p_pic_out)
    {
        msg_Err(p_filter, N_("Failed to allocate p_pic_out"));
        picture_Release(p_pic_in);
        return p_pic_in;
    }

    msg_Dbg(p_filter, N_("Before *p_pic_out = *p_pic_in"));
    *p_pic_out = *p_pic_in;
    msg_Dbg(p_filter, N_("After *p_pic_out = *p_pic_in"));
    
    if (p_pic_out->i_planes != 0) 
    {           
        u_int32_t *Y = p_pic_out->p[0].p_pixels;
        u_int32_t *U = p_pic_out->p[1].p_pixels;
        u_int32_t *V = p_pic_out->p[2].p_pixels;
        unsigned int width = p_pic_out->format.i_width;
        unsigned int heigth = p_pic_out->format.i_height;
        // stride
        unsigned int stride = p_pic_out->p[0].i_pitch;
        msg_Dbg(p_filter, "Stride: %d", stride);
        // cstride
        unsigned int cstride = p_pic_out->p[1].i_pitch;
        msg_Dbg(p_filter, "Cstride: %d", cstride);
        

        for (int y = 0; y < p_pic_out->p[0].i_visible_lines; y+=16)
        {
            msg_Dbg(p_filter, N_("Before calling add_grain, iteration y=%d"), y);
            ptr_add_grain(Y, U, V, y, width, heigth, stride, cstride);
            msg_Dbg(p_filter, N_("After calling add_grain, iteration y=%d"), y);
            Y += 16*stride*depth;
            U += 16*cstride*depth;
            V += 16*cstride*depth;
        }
        
        picture_CopyProperties(p_pic_out, p_pic_in);
        picture_Release(p_pic_in); //Decrement refs count. When it reaches 0, the picture is automatically destroyed (freed)
        return p_pic_out;
    }
    else
    {
        picture_Release(p_pic_out);
        return p_pic_in;
    }
    
}

static int FilterCallback(vlc_object_t *p_this, char const *psz_var,
                          vlc_value_t oldval, vlc_value_t newval,
                          void *p_data)
{
    VLC_UNUSED(oldval);
    my_filter_sys_t *p_sys = (my_filter_sys_t *)p_data;
    msg_Dbg(p_this, N_("Entered in callback"));
    vlc_mutex_lock(&p_sys->lock);

    if (!strcmp(FILTER_PREFIX "grain-amount", psz_var)) // Si la varaible modifiée est FILTER_PREFIX "grain-amount"
    {
        p_sys->i_grain_amount = VLC_CLIP(newval.i_int, 0, 255);
        msg_Dbg(p_this, "Changed value of grain-amount to %d", newval.i_int);
    }
    else if (!strcmp(FILTER_PREFIX "enabled", psz_var))
    {
        p_sys->b_enabled = newval.b_bool;
        msg_Dbg(p_this, "Changed value of enabled to %s", (newval.b_bool == true) ? "true" : "false");
    }
    vlc_mutex_unlock(&p_sys->lock);
}

vlc_module_begin()
    set_shortname(N_("InterGrain"))
    set_text_domain(DOMAIN)
    set_description(N_("InterDigital Grain Filter"))
    set_help(N_(HELP_STR))
    set_category(CAT_VIDEO)
    set_subcategory(SUBCAT_VIDEO_VFILTER)
    set_capability("video filter", 10)
    add_bool(FILTER_PREFIX "enabled", true, N_("Enable grain filter"), N_("Enable InterDigital grain filter"), true)
    add_integer_with_range(FILTER_PREFIX "grain-amount", 50, 0, 255, N_("Grain amount"), N_("Set a grain amount factor"), true)
    set_callbacks(Open, Close)
    add_shortcut("intergrain")
vlc_module_end()