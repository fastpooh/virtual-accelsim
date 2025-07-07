#ifndef TLB_H
#define TLB_H 

#include <vector>
#include <list>
#include <queue>
#include <cassert>
#include "../abstract_hardware_model.h"
#include "shader.h"

enum tlb_type {
    L1_i_TLB,
    L1_d_TLB,
    L2_u_TLB,
    L3_u_TLB,
    L4_u_TLB,
    NUM_TLB_TYPES
};

struct translation_request {
    translation_request(uint64_t time, mem_fetch* mf)
      : addr(mf->get_addr()), m_time_translation_start(time), m_mf(mf) {}

    uint64_t addr;
    uint64_t m_time_translation_start;
    uint64_t m_time_translation_done;
    mem_fetch* m_mf;

    void set_status(enum mem_fetch_status status, unsigned long long cycle)
    {
        m_mf->set_status(status, cycle);
    }
};

class tlb_t;
class tlb_hierarchy;
extern tlb_hierarchy *m_tlb_hierarchy;

struct tlb_entry_t {
    tlb_entry_t(unsigned num_sub_entries, tlb_hierarchy* tlb_hierarchy)
       : m_tlb_hierarchy(tlb_hierarchy) {}

    bool lookup(translation_request* tr, uint64_t time);
    void entry_flush();

    uint64_t page_num = 1;
    uint64_t accessed_time = 1;
    tlb_hierarchy* m_tlb_hierarchy;
};

// Set in TLB
class set_t {
    public:
      set_t(unsigned num_entries, unsigned num_sub_entries, tlb_t* tlb,
            tlb_hierarchy* tlb_hierarchy);

      void insert_tlb_entry(translation_request* tr, uint64_t time);
      bool lookup(translation_request* tr, uint64_t time);
      void print_tlb_set();
      void print_tlb_entry_time();
      bool check_page_exist(uint64_t page_num);
      void entry_flush();
  
    private:
      static unsigned m_next_set_id;
      const unsigned m_set_id;
      const unsigned m_num_entries;
      std::vector<tlb_entry_t> m_entries;
  
      tlb_t* m_tlb;
      tlb_hierarchy* m_tlb_hierarchy;

  };

// Type 1: tlb_t
class tlb_t
{
public:
    tlb_t(tlb_type type, unsigned tlb_id, tlb_hierarchy* tlb_hierarchy);

    virtual void push_translation_request(uint64_t time, translation_request* tr) = 0;
    virtual void cycle(uint64_t time) = 0;
    unsigned get_tlb_id();
    tlb_type get_tlb_type();
    unsigned get_translation_queue_size();


protected:
    const tlb_type m_type;
    const unsigned m_tlb_id;
    unsigned m_access_latency;

    std::queue<translation_request*> m_translation_queue;

    tlb_hierarchy* m_tlb_hierarchy;
    gpgpu_sim* m_gpu;
};

// Type 2: u_tlb_t
class u_tlb_t : public tlb_t {
    public:
      u_tlb_t(tlb_type type, unsigned tlb_id, tlb_hierarchy* tlb_hierarchy);

      void push_translation_request(uint64_t time, translation_request* tr) override;
      void cycle(uint64_t time) override;

      void access(uint64_t time);
      void issue_missed_translations(uint64_t time);
      void commit(uint64_t time);
      unsigned get_set(uint64_t addr);
      void fill(translation_request* tr, uint64_t time);
      bool check_merged_page_exist(uint64_t page_num, uint64_t time);
      void print_tlb_info();
      void print_tlb_info_detail();
      bool has_page_in_tlb(uint64_t page_num);
      void print_all_for_debugging(uint64_t time);
      void print_merged_pages_list();
      void push_to_complete_queue(translation_request* tr);
      void tlb_flush();
  
    protected:
      unsigned m_num_sets;
      unsigned m_num_ways;
      unsigned m_num_sub_entries;
  
      unsigned m_set_shift;
      unsigned m_sub_entry_shift;
  
      std::vector<set_t*> m_sets;
      std::list<translation_request*> m_completed_translations;             // These will enter higher TLB level or icnt
      std::queue<translation_request*> m_pending_missed_translations;       // These will enter lower TLB level or PTW

      std::queue<translation_request*> m_merged_translations;               // These wait in the queue until previous same TLB request enters
      std::list<uint64_t> m_merged_pages;
  };


// Type 3: d_tlb_t
class d_tlb_t : public u_tlb_t
{
public:
    d_tlb_t(tlb_type type, unsigned tlb_id, tlb_hierarchy* tlb_hierarchy);
    void cycle(uint64_t time) override;

private:
    bool m_stalled;
};


class tlb_hierarchy
{
public:
    tlb_hierarchy(const shader_core_config *shader_config, class gpgpu_sim *gpu);

    void push_translation_request(uint64_t time, mem_fetch* mf);
    void cycle(uint64_t time);
    void commit_translation(uint64_t time, tlb_type type, translation_request* tr);
    bool can_commit_translation(tlb_type type, translation_request* tr);
    bool tlb_miss(uint64_t time, tlb_type type, unsigned tlb_id, translation_request* tr);
    void print_tlb_stat();
    void tlb_flush();

private:
    // const tlb_config *m_config;
    const shader_core_config * m_shader_config;

    // L1 TLB
    unsigned m_n_l1_tlb;
    std::vector<d_tlb_t*> m_l1_d_tlb;

    // L2 TLB
    unsigned m_n_l2_tlb;
    std::vector<u_tlb_t*> m_l2_u_tlb;

    // L3 TLB
    u_tlb_t* m_l3_u_tlb;
    
    // L4 TLB (Page Table Walk)
    u_tlb_t* m_l4_u_tlb;

    class gpgpu_sim *m_gpu;
};

#endif