#ifndef PLUGIN_H
#define PLUGIN_H

typedef struct
{
    int i_grain_amount;
    bool b_enabled;
    vlc_mutex_t lock;
} my_filter_sys_t;

static int Open(vlc_object_t *p_this);
static void Close(vlc_object_t *p_this);
static picture_t *Filter(filter_t *p_filter, picture_t *p_pic_in);
static int FilterCallback( vlc_object_t *p_this, char const *psz_var,
                               vlc_value_t oldval, vlc_value_t newval,
                               void *p_data );
static int supportedChroma(vlc_fourcc_t *i_chroma);

#endif